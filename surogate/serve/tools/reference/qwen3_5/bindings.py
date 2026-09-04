"""Typed binding of a generic SInfer artifact to the hybrid GDN/attention target.

The converter and this module deliberately implement the target contract on
opposite sides of the artifact boundary.  Binding resolves persistent names
once, validates the complete target inventory, and exposes only typed block
and view objects to the model hot path.

One reference serves every size of the architecture, so the expected inventory is
a function of the artifact's own geometry rather than a compiled list, and the
shape of that inventory follows the objects the artifact actually stores: which
projections a checkpoint fuses, and whether it ships a vision tower, are read
from the object names, never inferred from a dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import torch

from surogate.serve.artifact import (
    Artifact,
    ArtifactIdentity,
    ResourceObject,
    TensorObject,
    decode_direct,
)

from .config import (
    ModelConfig,
    VisionConfig,
    model_config_from_declared,
    vision_config_for,
    vision_config_from_declared,
)


WEIGHTS_ID = "groupwise-int"

CONTIGUOUS = "contiguous-le-v1"
ROW_SPLIT = "row-split-k128-v1"
RESOURCE_ENCODING = "raw-bytes-v1"

BF16 = "BF16"
FP32 = "FP32"
I32 = "I32"
Q4 = "Q4G64_F16S"
Q5 = "Q5G64_F16S"
Q6 = "Q6G64_F16S"
W8 = "W8G32_F16S"

#: The group-wise types a linear matrix may be stored at.  Which one a given matrix
#: uses is the converter's choice per checkpoint, so a bound weight takes the format
#: the artifact declares and only its extent is contract.
LINEAR_FORMATS = frozenset((Q4, Q5, Q6, W8))

Component = Literal["text", "draft", "mtp", "vision"]


class BindingError(ValueError):
    """The generic artifact does not implement this registered target."""


@dataclass(frozen=True, slots=True)
class PhysicalBlock:
    """One stored tensor after its target role has been resolved."""

    tensor_id: int
    descriptor: TensorObject
    component: Component

    @property
    def shape(self) -> tuple[int, ...]:
        return self.descriptor.shape

    @property
    def format(self) -> str:
        return self.descriptor.format

    @property
    def layout(self) -> str:
        return self.descriptor.layout

    @property
    def payload_bytes(self) -> int:
        return self.descriptor.bytes


@dataclass(frozen=True, slots=True)
class LogicalRowView:
    """A consecutive logical row interval within a row-split block."""

    block: PhysicalBlock
    row_begin: int
    row_count: int
    shape: tuple[int, int]

    @property
    def row_end(self) -> int:
        return self.row_begin + self.row_count


@dataclass(frozen=True, slots=True)
class AxisView:
    """A logical axis order over a physical block."""

    block: PhysicalBlock
    axes: tuple[int, ...]
    shape: tuple[int, ...]


WeightObject: TypeAlias = PhysicalBlock | LogicalRowView | AxisView
RowAddressable: TypeAlias = PhysicalBlock | LogicalRowView


@dataclass(frozen=True, slots=True)
class BoundResource:
    descriptor: ResourceObject


@dataclass(frozen=True, slots=True)
class FrontendResources:
    tokenizer_json: BoundResource
    tokenizer_config_json: BoundResource
    chat_template_jinja: BoundResource
    generation_config_json: BoundResource
    preprocessor_config_json: BoundResource
    video_preprocessor_config_json: BoundResource


@dataclass(frozen=True, slots=True)
class MlpBinding:
    gate_up: PhysicalBlock
    gate: LogicalRowView
    up: LogicalRowView
    down: PhysicalBlock


@dataclass(frozen=True, slots=True)
class FullAttentionBinding:
    """A full-attention layer's weights, whichever way the checkpoint fuses them.

    `query_key` and `gate_value` are stored objects where the checkpoint splits them
    and row intervals of the one fused parent where it does not; the four leaf views
    address the same rows either way.
    """

    query_key: RowAddressable
    query: LogicalRowView
    key: LogicalRowView
    gate_value: RowAddressable
    output_gate: LogicalRowView
    value: LogicalRowView
    query_norm: PhysicalBlock
    key_norm: PhysicalBlock
    output: PhysicalBlock


@dataclass(frozen=True, slots=True)
class GdnBinding:
    """A gated-delta-net layer's weights, whichever way the checkpoint fuses them.

    `value_z` is the fused value|z parent, which only the layouts that store one
    expose; `value` and `z` are bound in every layout.
    """

    a_log: PhysicalBlock
    dt_bias: PhysicalBlock
    convolution_storage: PhysicalBlock
    convolution: AxisView
    a_projection: PhysicalBlock
    b_projection: PhysicalBlock
    query_key: RowAddressable
    query: LogicalRowView
    key: LogicalRowView
    value_z: RowAddressable | None
    value: LogicalRowView
    norm: PhysicalBlock
    z: RowAddressable
    output: PhysicalBlock


@dataclass(frozen=True, slots=True)
class TextLayerBinding:
    index: int
    input_norm: PhysicalBlock
    attention: FullAttentionBinding | None
    gdn: GdnBinding | None
    post_attention_norm: PhysicalBlock
    mlp: MlpBinding


@dataclass(frozen=True, slots=True)
class DraftHeadBinding:
    weight: PhysicalBlock
    token_ids: PhysicalBlock


@dataclass(frozen=True, slots=True)
class TextBinding:
    token_embedding: PhysicalBlock
    layers: tuple[TextLayerBinding, ...]
    final_norm: PhysicalBlock
    output_head: PhysicalBlock
    draft_head: DraftHeadBinding


@dataclass(frozen=True, slots=True)
class MtpAttentionBinding:
    query_key_gate_value: PhysicalBlock
    query: LogicalRowView
    key: LogicalRowView
    output_gate: LogicalRowView
    value: LogicalRowView
    query_norm: PhysicalBlock
    key_norm: PhysicalBlock
    output: PhysicalBlock


@dataclass(frozen=True, slots=True)
class MtpLayerBinding:
    input_norm: PhysicalBlock
    attention: MtpAttentionBinding
    post_attention_norm: PhysicalBlock
    mlp: MlpBinding


@dataclass(frozen=True, slots=True)
class MtpBinding:
    # These three fields are target aliases, not additional stored objects.
    token_embedding: PhysicalBlock
    full_output_head: PhysicalBlock
    optimized_proposal_head: DraftHeadBinding
    input_projection: PhysicalBlock
    embedding_norm: PhysicalBlock
    hidden_norm: PhysicalBlock
    layer: MtpLayerBinding
    final_norm: PhysicalBlock


@dataclass(frozen=True, slots=True)
class VisionLayerBinding:
    index: int
    attention_qkv: PhysicalBlock
    attention_qkv_bias: PhysicalBlock
    attention_output: PhysicalBlock
    attention_output_bias: PhysicalBlock
    mlp_fc1: PhysicalBlock
    mlp_fc1_bias: PhysicalBlock
    mlp_fc2: PhysicalBlock
    mlp_fc2_bias: PhysicalBlock
    norm1_weight: PhysicalBlock
    norm1_bias: PhysicalBlock
    norm2_weight: PhysicalBlock
    norm2_bias: PhysicalBlock


@dataclass(frozen=True, slots=True)
class VisionMergerBinding:
    fc1: PhysicalBlock
    fc1_bias: PhysicalBlock
    fc2: PhysicalBlock
    fc2_bias: PhysicalBlock
    norm_weight: PhysicalBlock
    norm_bias: PhysicalBlock


@dataclass(frozen=True, slots=True)
class VisionBinding:
    patch_embedding: PhysicalBlock
    patch_embedding_bias: PhysicalBlock
    position_embedding: PhysicalBlock
    layers: tuple[VisionLayerBinding, ...]
    merger: VisionMergerBinding


@dataclass(frozen=True, slots=True)
class _ExpectedTensor:
    name: str
    shape: tuple[int, ...]
    #: `None` where the artifact chooses the group-wise type; see LINEAR_FORMATS.
    format: str | None
    layout: str


@dataclass(frozen=True, slots=True)
class _ExpectedResource:
    name: str
    encoding: str = RESOURCE_ENCODING


_ExpectedObject: TypeAlias = _ExpectedTensor | _ExpectedResource


def _tensor(name: str, shape: tuple[int, ...], format_name: str) -> _ExpectedTensor:
    """A direct control tensor, at the one format its role requires."""

    return _ExpectedTensor(name, shape, format_name, CONTIGUOUS)


def _linear(name: str, shape: tuple[int, int]) -> _ExpectedTensor:
    """A matrix at whatever group-wise type the artifact declares for it."""

    return _ExpectedTensor(name, shape, None, ROW_SPLIT)


@dataclass(frozen=True, slots=True)
class Layout:
    """Which fused parents this artifact stores, read from its object names."""

    split_attention: bool
    gdn_split_qk_vz: bool
    gdn_split_qkv_z: bool
    vision: bool

    @property
    def gdn_fused(self) -> bool:
        return not (self.gdn_split_qk_vz or self.gdn_split_qkv_z)


def _layout_of(names: frozenset[str], cfg: ModelConfig) -> Layout:
    first_full = next(
        (layer for layer in range(cfg.layers) if cfg.is_full(layer)), None
    )
    first_gdn = next(
        (layer for layer in range(cfg.layers) if not cfg.is_full(layer)), None
    )
    if first_full is None or first_gdn is None:
        raise BindingError("a hybrid decoder needs both attention and GDN layers")
    full_prefix = f"text/layers/{first_full}/"
    gdn_prefix = f"text/layers/{first_gdn}/"
    return Layout(
        split_attention=full_prefix + "attention/query_key" in names,
        gdn_split_qk_vz=gdn_prefix + "gdn/query_key" in names,
        gdn_split_qkv_z=gdn_prefix + "gdn/query_key_value" in names,
        vision="vision/patch_embedding" in names,
    )


def _geometry(
    shapes: dict[str, tuple[int, ...]],
    declared: dict[str, float] | None,
) -> ModelConfig:
    """The decoder's dimensions, resolved tensor by tensor.

    Several dimensions coincide at some sizes -- at the 0.8B the attention query
    projection, the GDN key projection and the GDN value projection are all 2048
    rows wide -- so each is read from the one tensor whose role defines it, never
    matched by value.  Whatever no tensor pins down (the token domain, the rotary
    width, the epsilons) comes from the family default with the artifact's own
    `geometry` declaration laid over it.
    """

    base = model_config_from_declared(declared)

    def shape(name: str) -> tuple[int, ...]:
        found = shapes.get(name)
        if found is None:
            raise BindingError(f"artifact is missing the required object {name!r}")
        return found

    def exact_division(value: int, divisor: int, label: str) -> int:
        if divisor <= 0 or value % divisor:
            raise BindingError(f"{label}: {value} is not a multiple of {divisor}")
        return value // divisor

    # The token table is [vocab, hidden]; the residual stream's width is its column count.
    embedding = shape("text/token_embedding")
    if len(embedding) != 2:
        raise BindingError("text/token_embedding must be rank two")
    vocab, hidden = embedding

    layers = 0
    while f"text/layers/{layers}/input_norm" in shapes:
        layers += 1
    if layers == 0:
        raise BindingError("artifact declares no text layers")

    full = [layer for layer in range(layers) if f"text/layers/{layer}/attention/output" in shapes]
    gdn = [layer for layer in range(layers) if f"text/layers/{layer}/gdn/output" in shapes]
    if not full or not gdn or len(full) + len(gdn) != layers:
        raise BindingError("every text layer must be either full attention or GDN")
    # Attention lands on every `full_interval`-th layer, counting from one.
    full_interval = full[0] + 1
    if any(layer + 1 != (index + 1) * full_interval for index, layer in enumerate(full)):
        raise BindingError("full-attention layers are not evenly spaced")

    full_prefix = f"text/layers/{full[0]}/"
    gdn_prefix = f"text/layers/{gdn[0]}/"

    # The per-head RMS norms are [head_dim]; the attention output projection is
    # [hidden, query_size], which is what fixes the query head count.
    (head_dim,) = shape(full_prefix + "attention/query_norm")
    q_size = shape(full_prefix + "attention/output")[1]
    q_heads = exact_division(q_size, head_dim, "attention query rows")

    # The attention input parent holds query|key once when split and query|key|gate|value
    # when fused, so the key rows follow from the parent this artifact actually stores.
    if full_prefix + "attention/query_key" in shapes:
        attention_input_rows = shape(full_prefix + "attention/query_key")[0]
        kv_size = attention_input_rows - q_size
    else:
        attention_input_rows = shape(full_prefix + "attention/query_key_gate_value")[0]
        kv_size = exact_division(attention_input_rows, 2, "fused attention rows") - q_size
    kv_heads = exact_division(kv_size, head_dim, "attention key/value rows")

    # `a_log` carries one gate per GDN value head; `gdn/norm` is one key head wide;
    # the GDN output projection is [hidden, value_dim] -- the value width, not the
    # query width, even where the two happen to be equal.
    (gdn_v_heads,) = shape(gdn_prefix + "gdn/a_log")
    (gdn_k_dim,) = shape(gdn_prefix + "gdn/norm")
    value_dim = shape(gdn_prefix + "gdn/output")[1]
    gdn_v_dim = exact_division(value_dim, gdn_v_heads, "GDN value rows")

    # The convolution is [width, 2 * key_dim + value_dim] over the query, key and
    # value channels it smooths, which is what separates the key width from the rest.
    conv = shape(gdn_prefix + "gdn/convolution")
    if len(conv) != 2:
        raise BindingError("gdn/convolution must be rank two")
    conv_width, conv_dim = conv
    key_dim = exact_division(conv_dim - value_dim, 2, "GDN convolution channels")
    gdn_k_heads = exact_division(key_dim, gdn_k_dim, "GDN key rows")

    # The MLP down projection is [hidden, intermediate]; the gate|up parent above it
    # is twice as tall, and the contract checks that.
    intermediate = shape(gdn_prefix + "mlp/down")[1]

    return ModelConfig(
        hidden=hidden,
        layers=layers,
        intermediate=intermediate,
        vocab=vocab,
        token_domain=base.token_domain,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        rotary_dim=base.rotary_dim,
        full_interval=full_interval,
        gdn_k_heads=gdn_k_heads,
        gdn_v_heads=gdn_v_heads,
        gdn_k_dim=gdn_k_dim,
        gdn_v_dim=gdn_v_dim,
        conv_width=conv_width,
        rms_eps=base.rms_eps,
        rope_theta=base.rope_theta,
        mrope_section=base.mrope_section,
        prefill_chunk=base.prefill_chunk,
        max_position_embeddings=base.max_position_embeddings,
    )


def _vision_geometry(
    shapes: dict[str, tuple[int, ...]],
    declared: dict[str, float] | None,
    cfg: ModelConfig,
) -> VisionConfig:
    """The tower's dimensions, resolved tensor by tensor."""

    base = vision_config_for(cfg, vision_config_from_declared(declared))

    def shape(name: str) -> tuple[int, ...]:
        found = shapes.get(name)
        if found is None:
            raise BindingError(f"artifact is missing the required object {name!r}")
        return found

    # The patch embedding is [hidden, patch_dim]: the tower's width and the flattened
    # pixel volume of one patch.
    patch_embedding = shape("vision/patch_embedding")
    if len(patch_embedding) != 2:
        raise BindingError("vision/patch_embedding must be rank two")
    hidden, patch_dim = patch_embedding
    if patch_dim != base.patch_dim:
        raise BindingError(
            f"vision patch volume is {patch_dim}; the declared patch geometry gives "
            f"{base.patch_dim}"
        )

    depth = 0
    while f"vision/layers/{depth}/norm1/weight" in shapes:
        depth += 1
    if depth == 0:
        raise BindingError("artifact declares a vision tower with no layers")

    intermediate = shape("vision/layers/0/mlp/fc1")[0]
    (position_embeddings, _) = shape("vision/position_embedding")

    # The merger consumes one merged block of patches at a time, so its input width is
    # `hidden * merge * merge`; its output width is the text model's hidden state.
    merger_hidden = shape("vision/merger/fc1")[0]
    merge_unit, remainder = divmod(merger_hidden, hidden)
    side = int(round(merge_unit**0.5))
    if remainder or side * side != merge_unit:
        raise BindingError(
            f"vision merger input {merger_hidden} is not a square block of {hidden}-wide patches"
        )
    out_hidden = shape("vision/merger/fc2")[0]

    return VisionConfig(
        depth=depth,
        hidden=hidden,
        intermediate=intermediate,
        out_hidden=out_hidden,
        heads=base.heads,
        in_channels=base.in_channels,
        patch=base.patch,
        temporal_patch=base.temporal_patch,
        spatial_merge=side,
        position_embeddings=position_embeddings,
        rope_theta=base.rope_theta,
        norm_eps=base.norm_eps,
    )


def _text_contract(
    cfg: ModelConfig, layout: Layout, draft_vocab: int
) -> tuple[_ExpectedTensor, ...]:
    tensors: list[_ExpectedTensor] = [
        _linear("text/token_embedding", (cfg.vocab, cfg.hidden))
    ]
    for layer in range(cfg.layers):
        prefix = f"text/layers/{layer}/"
        tensors.append(_tensor(prefix + "input_norm", (cfg.hidden,), BF16))
        if cfg.is_full(layer):
            attention_pair = (cfg.q_size + cfg.kv_size, cfg.hidden)
            if layout.split_attention:
                tensors.extend(
                    (
                        _linear(prefix + "attention/query_key", attention_pair),
                        _linear(prefix + "attention/gate_value", attention_pair),
                    )
                )
            else:
                tensors.append(
                    _linear(
                        prefix + "attention/query_key_gate_value",
                        (cfg.attention_input_rows, cfg.hidden),
                    )
                )
            tensors.extend(
                (
                    _tensor(prefix + "attention/query_norm", (cfg.head_dim,), BF16),
                    _tensor(prefix + "attention/key_norm", (cfg.head_dim,), BF16),
                    _linear(prefix + "attention/output", (cfg.hidden, cfg.q_size)),
                )
            )
        else:
            tensors.extend(
                (
                    _tensor(prefix + "gdn/a_log", (cfg.gdn_v_heads,), FP32),
                    _tensor(prefix + "gdn/dt_bias", (cfg.gdn_v_heads,), FP32),
                    _tensor(
                        prefix + "gdn/convolution", (cfg.conv_width, cfg.conv_dim), BF16
                    ),
                    _tensor(
                        prefix + "gdn/a_projection", (cfg.gdn_v_heads, cfg.hidden), BF16
                    ),
                    _tensor(
                        prefix + "gdn/b_projection", (cfg.gdn_v_heads, cfg.hidden), BF16
                    ),
                )
            )
            if layout.gdn_split_qk_vz:
                tensors.extend(
                    (
                        _linear(prefix + "gdn/query_key", (2 * cfg.key_dim, cfg.hidden)),
                        _linear(prefix + "gdn/value_z", (2 * cfg.value_dim, cfg.hidden)),
                    )
                )
            elif layout.gdn_split_qkv_z:
                tensors.extend(
                    (
                        _linear(
                            prefix + "gdn/query_key_value", (cfg.conv_dim, cfg.hidden)
                        ),
                        _linear(prefix + "gdn/z", (cfg.value_dim, cfg.hidden)),
                    )
                )
            else:
                tensors.append(
                    _linear(
                        prefix + "gdn/query_key_value_z",
                        (cfg.conv_dim + cfg.value_dim, cfg.hidden),
                    )
                )
            tensors.extend(
                (
                    _tensor(prefix + "gdn/norm", (cfg.gdn_k_dim,), BF16),
                    _linear(prefix + "gdn/output", (cfg.hidden, cfg.value_dim)),
                )
            )
        tensors.extend(
            (
                _tensor(prefix + "post_attention_norm", (cfg.hidden,), BF16),
                _linear(prefix + "mlp/gate_up", (2 * cfg.intermediate, cfg.hidden)),
                _linear(prefix + "mlp/down", (cfg.hidden, cfg.intermediate)),
            )
        )
    tensors.extend(
        (
            _tensor("text/final_norm", (cfg.hidden,), BF16),
            _linear("text/output_head", (cfg.vocab, cfg.hidden)),
            _linear("text/draft_head", (draft_vocab, cfg.hidden)),
            _tensor("text/draft_head_token_ids", (draft_vocab,), I32),
        )
    )
    return tuple(tensors)


def _mtp_contract(cfg: ModelConfig) -> tuple[_ExpectedTensor, ...]:
    return (
        # The MTP block projects the concatenated token embedding and hidden state,
        # so its input is twice the residual width -- not the query width, which is
        # the same number at some sizes.
        _linear("mtp/input_projection", (cfg.hidden, cfg.mtp_input_rows)),
        _tensor("mtp/embedding_norm", (cfg.hidden,), BF16),
        _tensor("mtp/hidden_norm", (cfg.hidden,), BF16),
        _tensor("mtp/layer/input_norm", (cfg.hidden,), BF16),
        _linear(
            "mtp/layer/attention/query_key_gate_value",
            (cfg.attention_input_rows, cfg.hidden),
        ),
        _tensor("mtp/layer/attention/query_norm", (cfg.head_dim,), BF16),
        _tensor("mtp/layer/attention/key_norm", (cfg.head_dim,), BF16),
        _linear("mtp/layer/attention/output", (cfg.hidden, cfg.q_size)),
        _tensor("mtp/layer/post_attention_norm", (cfg.hidden,), BF16),
        _linear("mtp/layer/mlp/gate_up", (2 * cfg.intermediate, cfg.hidden)),
        _linear("mtp/layer/mlp/down", (cfg.hidden, cfg.intermediate)),
        _tensor("mtp/final_norm", (cfg.hidden,), BF16),
    )


def _vision_contract(vision: VisionConfig | None) -> tuple[_ExpectedTensor, ...]:
    if vision is None:
        return ()
    hidden = vision.hidden
    tensors: list[_ExpectedTensor] = [
        _linear("vision/patch_embedding", (hidden, vision.patch_dim)),
        _tensor("vision/patch_embedding_bias", (hidden,), BF16),
        _tensor(
            "vision/position_embedding", (vision.position_embeddings, hidden), BF16
        ),
    ]
    for layer in range(vision.depth):
        prefix = f"vision/layers/{layer}/"
        tensors.extend(
            (
                _linear(prefix + "attention/qkv", (3 * hidden, hidden)),
                _tensor(prefix + "attention/qkv_bias", (3 * hidden,), BF16),
                _linear(prefix + "attention/output", (hidden, hidden)),
                _tensor(prefix + "attention/output_bias", (hidden,), BF16),
                _linear(prefix + "mlp/fc1", (vision.intermediate, hidden)),
                _tensor(prefix + "mlp/fc1_bias", (vision.intermediate,), BF16),
                _linear(prefix + "mlp/fc2", (hidden, vision.intermediate)),
                _tensor(prefix + "mlp/fc2_bias", (hidden,), BF16),
                _tensor(prefix + "norm1/weight", (hidden,), BF16),
                _tensor(prefix + "norm1/bias", (hidden,), BF16),
                _tensor(prefix + "norm2/weight", (hidden,), BF16),
                _tensor(prefix + "norm2/bias", (hidden,), BF16),
            )
        )
    merger = vision.merger_hidden
    tensors.extend(
        (
            _linear("vision/merger/fc1", (merger, merger)),
            _tensor("vision/merger/fc1_bias", (merger,), BF16),
            _linear("vision/merger/fc2", (vision.out_hidden, merger)),
            _tensor("vision/merger/fc2_bias", (vision.out_hidden,), BF16),
            _tensor("vision/merger/norm/weight", (hidden,), BF16),
            _tensor("vision/merger/norm/bias", (hidden,), BF16),
        )
    )
    return tuple(tensors)


RESOURCE_CONTRACT = tuple(
    _ExpectedResource(name)
    for name in (
        "frontend/tokenizer.json",
        "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",
        "frontend/generation_config.json",
        "frontend/preprocessor_config.json",
        "frontend/video_preprocessor_config.json",
    )
)


def object_contract(
    cfg: ModelConfig,
    layout: Layout,
    draft_vocab: int,
    vision: VisionConfig | None,
) -> tuple[_ExpectedObject, ...]:
    """The complete inventory an artifact of this geometry and layout must contain."""

    return (
        RESOURCE_CONTRACT
        + _text_contract(cfg, layout, draft_vocab)
        + _mtp_contract(cfg)
        + _vision_contract(vision)
    )


def expected_row_views(cfg: ModelConfig, layout: Layout) -> int:
    """How many logical row intervals a complete binding of this layout carves.

    A fused parent needs one view per component plus one per fused half that the
    model addresses as a unit; a split parent is already that unit.
    """

    per_full = 4 if layout.split_attention else 6
    per_gdn = 6 if layout.gdn_fused else 4
    mtp_views = 6  # four attention components and the MLP's gate|up halves
    return (
        cfg.full_layers * per_full
        + cfg.gdn_layers * per_gdn
        + cfg.layers * 2
        + mtp_views
    )


def _component(name: str) -> Component:
    if name.startswith("vision/"):
        return "vision"
    if name.startswith("mtp/"):
        return "mtp"
    if name in ("text/draft_head", "text/draft_head_token_ids"):
        return "draft"
    return "text"


def _validate_inventory(
    artifact: Artifact, contract: tuple[_ExpectedObject, ...]
) -> None:
    if artifact.identity.weights_id != WEIGHTS_ID:
        raise BindingError(
            f"artifact stores {artifact.identity.weights_id!r} weights; this reference "
            f"reads {WEIGHTS_ID!r}"
        )
    if len(artifact.objects) != len(contract):
        raise BindingError(
            f"artifact has {len(artifact.objects)} objects; expected {len(contract)}"
        )
    expected_by_name = {obj.name: obj for obj in contract}
    actual_names = {obj.name for obj in artifact.objects}
    expected_names = frozenset(expected_by_name)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise BindingError(
            f"artifact object names differ; missing={missing!r}, extra={extra!r}"
        )
    for actual in artifact.objects:
        expected = expected_by_name[actual.name]
        if isinstance(expected, _ExpectedTensor):
            if not isinstance(actual, TensorObject):
                raise BindingError(f"object {actual.name!r} is not a tensor")
            format_name = expected.format
            if format_name is None:
                if actual.format not in LINEAR_FORMATS:
                    raise BindingError(
                        f"object {actual.name!r} is stored at {actual.format!r}; a "
                        f"matrix must use one of {sorted(LINEAR_FORMATS)}"
                    )
                format_name = actual.format
            signature = (expected.name, expected.shape, format_name, expected.layout)
            if (
                actual.name,
                actual.shape,
                actual.format,
                actual.layout,
            ) != signature:
                raise BindingError(
                    f"object {actual.name!r} does not match tensor signature {signature!r}"
                )
        else:
            signature = (expected.name, expected.encoding)
            if not isinstance(actual, ResourceObject) or (
                actual.name,
                actual.encoding,
            ) != signature:
                raise BindingError(
                    f"object {actual.name!r} does not match resource signature {signature!r}"
                )


def _row_view(
    block: PhysicalBlock,
    row_begin: int,
    row_end: int,
    shape: tuple[int, int],
    views: list[LogicalRowView],
) -> LogicalRowView:
    if block.layout != ROW_SPLIT or len(block.shape) != 2:
        raise BindingError("logical row view parent must be a rank-two row-split block")
    if row_begin < 0 or row_end > block.shape[0] or row_begin >= row_end:
        raise BindingError("logical row view is outside its physical block")
    expected_shape = (row_end - row_begin, block.shape[1])
    if shape != expected_shape:
        raise BindingError(
            f"logical row view shape is {shape}, expected {expected_shape}"
        )
    view = LogicalRowView(block, row_begin, row_end - row_begin, shape)
    views.append(view)
    return view


class ArtifactBinding:
    """Complete typed target binding over one open generic artifact."""

    def __init__(self, artifact: Artifact, *, owns_artifact: bool = False):
        shapes = {
            obj.name: obj.shape
            for obj in artifact.objects
            if isinstance(obj, TensorObject)
        }
        names = frozenset(shapes)
        declared = getattr(artifact, "geometry", None)
        self.config = _geometry(shapes, declared)
        cfg = self.config
        self.layout = _layout_of(names, cfg)
        draft_shape = shapes.get("text/draft_head")
        if draft_shape is None or len(draft_shape) != 2:
            raise BindingError("artifact is missing a rank-two text/draft_head")
        self.draft_vocab = draft_shape[0]
        self.vision_config = (
            _vision_geometry(shapes, getattr(artifact, "vision_geometry", None), cfg)
            if self.layout.vision
            else None
        )
        self.contract = object_contract(
            cfg, self.layout, self.draft_vocab, self.vision_config
        )
        _validate_inventory(artifact, self.contract)
        self._artifact = artifact
        self._owns_artifact = owns_artifact

        resources: dict[str, BoundResource] = {}
        blocks: dict[str, PhysicalBlock] = {}
        tensors: list[PhysicalBlock] = []
        for obj in artifact.objects:
            if isinstance(obj, ResourceObject):
                resources[obj.name] = BoundResource(obj)
            else:
                block = PhysicalBlock(len(tensors), obj, _component(obj.name))
                tensors.append(block)
                blocks[obj.name] = block

        self.tensors = tuple(tensors)
        self.frontend = FrontendResources(
            resources["frontend/tokenizer.json"],
            resources["frontend/tokenizer_config.json"],
            resources["frontend/chat_template.jinja"],
            resources["frontend/generation_config.json"],
            resources["frontend/preprocessor_config.json"],
            resources["frontend/video_preprocessor_config.json"],
        )

        row_views: list[LogicalRowView] = []
        axis_views: list[AxisView] = []
        layers: list[TextLayerBinding] = []
        hidden = cfg.hidden
        q_size, kv_size = cfg.q_size, cfg.kv_size
        key_dim, value_dim = cfg.key_dim, cfg.value_dim
        for layer in range(cfg.layers):
            prefix = f"text/layers/{layer}/"
            input_norm = blocks[prefix + "input_norm"]
            if cfg.is_full(layer):
                if self.layout.split_attention:
                    query_parent = blocks[prefix + "attention/query_key"]
                    gate_parent = blocks[prefix + "attention/gate_value"]
                    query_key: RowAddressable = query_parent
                    gate_value: RowAddressable = gate_parent
                    query_base = gate_base = 0
                else:
                    parent = blocks[prefix + "attention/query_key_gate_value"]
                    pair = q_size + kv_size
                    query_key = _row_view(parent, 0, pair, (pair, hidden), row_views)
                    gate_value = _row_view(
                        parent, pair, 2 * pair, (pair, hidden), row_views
                    )
                    query_parent = gate_parent = parent
                    query_base, gate_base = 0, pair
                attention = FullAttentionBinding(
                    query_key=query_key,
                    query=_row_view(
                        query_parent,
                        query_base,
                        query_base + q_size,
                        (q_size, hidden),
                        row_views,
                    ),
                    key=_row_view(
                        query_parent,
                        query_base + q_size,
                        query_base + q_size + kv_size,
                        (kv_size, hidden),
                        row_views,
                    ),
                    gate_value=gate_value,
                    output_gate=_row_view(
                        gate_parent,
                        gate_base,
                        gate_base + q_size,
                        (q_size, hidden),
                        row_views,
                    ),
                    value=_row_view(
                        gate_parent,
                        gate_base + q_size,
                        gate_base + q_size + kv_size,
                        (kv_size, hidden),
                        row_views,
                    ),
                    query_norm=blocks[prefix + "attention/query_norm"],
                    key_norm=blocks[prefix + "attention/key_norm"],
                    output=blocks[prefix + "attention/output"],
                )
                gdn = None
            else:
                # Every layout stores query|key contiguously, then value, then z; only
                # the object boundaries between them move.
                if self.layout.gdn_split_qk_vz:
                    qk_parent = blocks[prefix + "gdn/query_key"]
                    vz_parent = blocks[prefix + "gdn/value_z"]
                    qk_view: RowAddressable = qk_parent
                    vz_view: RowAddressable | None = vz_parent
                    qk_base = value_base = 0
                    z_parent, z_base = vz_parent, value_dim
                elif self.layout.gdn_split_qkv_z:
                    qkv_parent = blocks[prefix + "gdn/query_key_value"]
                    qk_parent = vz_parent = qkv_parent
                    qk_view = _row_view(
                        qkv_parent, 0, 2 * key_dim, (2 * key_dim, hidden), row_views
                    )
                    vz_view = None
                    qk_base, value_base = 0, 2 * key_dim
                    z_parent, z_base = blocks[prefix + "gdn/z"], None
                else:
                    parent = blocks[prefix + "gdn/query_key_value_z"]
                    qk_parent = vz_parent = parent
                    qk_view = _row_view(
                        parent, 0, 2 * key_dim, (2 * key_dim, hidden), row_views
                    )
                    vz_view = _row_view(
                        parent,
                        2 * key_dim,
                        2 * key_dim + 2 * value_dim,
                        (2 * value_dim, hidden),
                        row_views,
                    )
                    qk_base, value_base = 0, 2 * key_dim
                    z_parent, z_base = parent, 2 * key_dim + value_dim
                convolution_storage = blocks[prefix + "gdn/convolution"]
                convolution = AxisView(
                    convolution_storage, (1, 0), (cfg.conv_dim, cfg.conv_width)
                )
                axis_views.append(convolution)
                gdn = GdnBinding(
                    a_log=blocks[prefix + "gdn/a_log"],
                    dt_bias=blocks[prefix + "gdn/dt_bias"],
                    convolution_storage=convolution_storage,
                    convolution=convolution,
                    a_projection=blocks[prefix + "gdn/a_projection"],
                    b_projection=blocks[prefix + "gdn/b_projection"],
                    query_key=qk_view,
                    query=_row_view(
                        qk_parent,
                        qk_base,
                        qk_base + key_dim,
                        (key_dim, hidden),
                        row_views,
                    ),
                    key=_row_view(
                        qk_parent,
                        qk_base + key_dim,
                        qk_base + 2 * key_dim,
                        (key_dim, hidden),
                        row_views,
                    ),
                    value_z=vz_view,
                    value=_row_view(
                        vz_parent,
                        value_base,
                        value_base + value_dim,
                        (value_dim, hidden),
                        row_views,
                    ),
                    norm=blocks[prefix + "gdn/norm"],
                    z=(
                        z_parent
                        if z_base is None
                        else _row_view(
                            z_parent,
                            z_base,
                            z_base + value_dim,
                            (value_dim, hidden),
                            row_views,
                        )
                    ),
                    output=blocks[prefix + "gdn/output"],
                )
                attention = None

            gate_up = blocks[prefix + "mlp/gate_up"]
            layers.append(
                TextLayerBinding(
                    index=layer,
                    input_norm=input_norm,
                    attention=attention,
                    gdn=gdn,
                    post_attention_norm=blocks[prefix + "post_attention_norm"],
                    mlp=self._mlp(gate_up, blocks[prefix + "mlp/down"], row_views),
                )
            )

        draft_head = DraftHeadBinding(
            blocks["text/draft_head"], blocks["text/draft_head_token_ids"]
        )
        self.text = TextBinding(
            token_embedding=blocks["text/token_embedding"],
            layers=tuple(layers),
            final_norm=blocks["text/final_norm"],
            output_head=blocks["text/output_head"],
            draft_head=draft_head,
        )

        mtp_qkgv = blocks["mtp/layer/attention/query_key_gate_value"]
        self.mtp = MtpBinding(
            token_embedding=self.text.token_embedding,
            full_output_head=self.text.output_head,
            optimized_proposal_head=draft_head,
            input_projection=blocks["mtp/input_projection"],
            embedding_norm=blocks["mtp/embedding_norm"],
            hidden_norm=blocks["mtp/hidden_norm"],
            layer=MtpLayerBinding(
                input_norm=blocks["mtp/layer/input_norm"],
                attention=MtpAttentionBinding(
                    query_key_gate_value=mtp_qkgv,
                    query=_row_view(mtp_qkgv, 0, q_size, (q_size, hidden), row_views),
                    key=_row_view(
                        mtp_qkgv,
                        q_size,
                        q_size + kv_size,
                        (kv_size, hidden),
                        row_views,
                    ),
                    output_gate=_row_view(
                        mtp_qkgv,
                        q_size + kv_size,
                        2 * q_size + kv_size,
                        (q_size, hidden),
                        row_views,
                    ),
                    value=_row_view(
                        mtp_qkgv,
                        2 * q_size + kv_size,
                        cfg.attention_input_rows,
                        (kv_size, hidden),
                        row_views,
                    ),
                    query_norm=blocks["mtp/layer/attention/query_norm"],
                    key_norm=blocks["mtp/layer/attention/key_norm"],
                    output=blocks["mtp/layer/attention/output"],
                ),
                post_attention_norm=blocks["mtp/layer/post_attention_norm"],
                mlp=self._mlp(
                    blocks["mtp/layer/mlp/gate_up"],
                    blocks["mtp/layer/mlp/down"],
                    row_views,
                ),
            ),
            final_norm=blocks["mtp/final_norm"],
        )

        self.vision = (
            self._bind_vision(blocks, self.vision_config)
            if self.vision_config is not None
            else None
        )
        self.row_views = tuple(row_views)
        self.axis_views = tuple(axis_views)
        expected_tensors = len(self.contract) - len(RESOURCE_CONTRACT)
        if len(self.tensors) != expected_tensors:
            raise RuntimeError(
                f"incomplete typed binding: {len(self.tensors)} of "
                f"{expected_tensors} tensors"
            )
        expected_views = expected_row_views(cfg, self.layout)
        if len(self.row_views) != expected_views:
            raise RuntimeError(
                f"incomplete typed binding: {len(self.row_views)} of "
                f"{expected_views} row views"
            )
        if len(self.axis_views) != cfg.gdn_layers:
            raise RuntimeError("incomplete GDN axis binding")
        self._validate_aliases()
        self._validate_draft_ids()

    def _mlp(
        self,
        gate_up: PhysicalBlock,
        down: PhysicalBlock,
        row_views: list[LogicalRowView],
    ) -> MlpBinding:
        rows = self.config.intermediate
        hidden = self.config.hidden
        return MlpBinding(
            gate_up=gate_up,
            gate=_row_view(gate_up, 0, rows, (rows, hidden), row_views),
            up=_row_view(gate_up, rows, 2 * rows, (rows, hidden), row_views),
            down=down,
        )

    @staticmethod
    def _bind_vision(
        blocks: dict[str, PhysicalBlock], vision: VisionConfig
    ) -> VisionBinding:
        vision_layers: list[VisionLayerBinding] = []
        for layer in range(vision.depth):
            prefix = f"vision/layers/{layer}/"
            vision_layers.append(
                VisionLayerBinding(
                    index=layer,
                    attention_qkv=blocks[prefix + "attention/qkv"],
                    attention_qkv_bias=blocks[prefix + "attention/qkv_bias"],
                    attention_output=blocks[prefix + "attention/output"],
                    attention_output_bias=blocks[prefix + "attention/output_bias"],
                    mlp_fc1=blocks[prefix + "mlp/fc1"],
                    mlp_fc1_bias=blocks[prefix + "mlp/fc1_bias"],
                    mlp_fc2=blocks[prefix + "mlp/fc2"],
                    mlp_fc2_bias=blocks[prefix + "mlp/fc2_bias"],
                    norm1_weight=blocks[prefix + "norm1/weight"],
                    norm1_bias=blocks[prefix + "norm1/bias"],
                    norm2_weight=blocks[prefix + "norm2/weight"],
                    norm2_bias=blocks[prefix + "norm2/bias"],
                )
            )
        return VisionBinding(
            patch_embedding=blocks["vision/patch_embedding"],
            patch_embedding_bias=blocks["vision/patch_embedding_bias"],
            position_embedding=blocks["vision/position_embedding"],
            layers=tuple(vision_layers),
            merger=VisionMergerBinding(
                fc1=blocks["vision/merger/fc1"],
                fc1_bias=blocks["vision/merger/fc1_bias"],
                fc2=blocks["vision/merger/fc2"],
                fc2_bias=blocks["vision/merger/fc2_bias"],
                norm_weight=blocks["vision/merger/norm/weight"],
                norm_bias=blocks["vision/merger/norm/bias"],
            ),
        )

    @classmethod
    def open(cls, path: str | Path) -> "ArtifactBinding":
        artifact = Artifact.open(path)
        try:
            return cls(artifact, owns_artifact=True)
        except BaseException:
            artifact.close()
            raise

    @classmethod
    def bind(cls, artifact: Artifact) -> "ArtifactBinding":
        return cls(artifact, owns_artifact=False)

    @property
    def identity(self) -> ArtifactIdentity:
        return self._artifact.identity

    def payload(self, block: PhysicalBlock) -> memoryview:
        return self._artifact.payload(block.descriptor)

    def resource_bytes(self, resource: BoundResource) -> bytes:
        return bytes(self._artifact.payload(resource.descriptor))

    def blocks_for(self, *components: Component) -> tuple[PhysicalBlock, ...]:
        wanted = frozenset(components)
        return tuple(block for block in self.tensors if block.component in wanted)

    def _validate_aliases(self) -> None:
        if self.mtp.token_embedding is not self.text.token_embedding:
            raise RuntimeError("MTP embedding alias is not bound to the text block")
        if self.mtp.full_output_head is not self.text.output_head:
            raise RuntimeError("MTP output-head alias is not bound to the text block")
        if self.mtp.optimized_proposal_head is not self.text.draft_head:
            raise RuntimeError("MTP proposal-head alias is not bound to the draft pair")
        expected = (self.config.conv_dim, self.config.conv_width)
        for layer in self.text.layers:
            if layer.gdn is None:
                continue
            view = layer.gdn.convolution
            if (
                view.block is not layer.gdn.convolution_storage
                or view.axes != (1, 0)
                or view.shape != expected
            ):
                raise RuntimeError("GDN channel-major convolution alias is incorrect")

    def _validate_draft_ids(self) -> None:
        block = self.text.draft_head.token_ids
        token_ids = decode_direct(
            self.payload(block), block.format, block.shape, device="cpu"
        )
        domain = self.config.token_domain
        if token_ids.dtype != torch.int32 or tuple(token_ids.shape) != (
            self.draft_vocab,
        ):
            raise BindingError(f"draft token IDs must be I32[{self.draft_vocab}]")
        if int(token_ids.min()) < 0 or int(token_ids.max()) >= domain:
            raise BindingError(f"draft token IDs are outside 0..{domain - 1}")
        if torch.unique(token_ids).numel() != token_ids.numel():
            raise BindingError("draft token IDs are not unique")

    def close(self) -> None:
        if self._owns_artifact:
            self._artifact.close()

    def __enter__(self) -> "ArtifactBinding":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "ArtifactBinding",
    "AxisView",
    "BindingError",
    "BoundResource",
    "DraftHeadBinding",
    "FrontendResources",
    "FullAttentionBinding",
    "GdnBinding",
    "Layout",
    "LogicalRowView",
    "MlpBinding",
    "MtpAttentionBinding",
    "MtpBinding",
    "MtpLayerBinding",
    "PhysicalBlock",
    "RowAddressable",
    "TextBinding",
    "TextLayerBinding",
    "VisionBinding",
    "VisionLayerBinding",
    "VisionMergerBinding",
    "WeightObject",
]
