"""Typed hybrid MoE bindings derived from complete artifact configuration."""

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


from types import SimpleNamespace
from .config import model_config_from_declared, vision_config_from_declared
from ..qwen3_5.bindings import _vision_contract as dense_vision_contract

LINEAR_FORMATS = {"BF16", "Q4G64_F16S", "Q5G64_F16S", "Q6G64_F16S", "W8G32_F16S"}

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

Component = Literal["text", "draft", "mtp", "vision", "dflash"]


class BindingError(ValueError):
    """The generic artifact does not implement this exact target profile."""


@dataclass(frozen=True, slots=True)
class PhysicalBlock:
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
    """A consecutive logical row interval within a rank-two parent."""

    block: PhysicalBlock
    row_begin: int
    row_count: int
    shape: tuple[int, int]

    @property
    def row_end(self) -> int:
        return self.row_begin + self.row_count


@dataclass(frozen=True, slots=True)
class AxisView:
    block: PhysicalBlock
    axes: tuple[int, ...]
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExpertBank:
    """One equal-stride expert bank addressed only for selected expert ids."""

    block: PhysicalBlock
    experts: int
    rows_per_expert: int
    split_rows: int | None


WeightObject: TypeAlias = PhysicalBlock | LogicalRowView | AxisView
RowAddressable: TypeAlias = PhysicalBlock | LogicalRowView


@dataclass(frozen=True, slots=True)
class BoundResource:
    descriptor: ResourceObject


@dataclass(frozen=True, slots=True)
class FrontendResources:
    tokenizer_json: BoundResource
    tokenizer_config_json: BoundResource
    chat_template_jinja: BoundResource | None
    generation_config_json: BoundResource
    preprocessor_config_json: BoundResource | None
    video_preprocessor_config_json: BoundResource | None


@dataclass(frozen=True, slots=True)
class MoeBinding:
    router_shared_gate: PhysicalBlock
    router: LogicalRowView
    shared_gate: LogicalRowView
    routed_gate_up: ExpertBank
    routed_down: ExpertBank
    shared_gate_up: PhysicalBlock
    shared_expert_gate: LogicalRowView
    shared_up: LogicalRowView
    shared_down: PhysicalBlock


@dataclass(frozen=True, slots=True)
class FullAttentionBinding:
    query_key_gate_value: PhysicalBlock
    query: LogicalRowView
    key: LogicalRowView
    output_gate: LogicalRowView
    value: LogicalRowView
    query_norm: PhysicalBlock
    key_norm: PhysicalBlock
    output: PhysicalBlock


@dataclass(frozen=True, slots=True)
class GdnBinding:
    a_log: PhysicalBlock
    dt_bias: PhysicalBlock
    convolution_storage: PhysicalBlock
    convolution: AxisView
    a_b_projection: PhysicalBlock
    a_projection: LogicalRowView
    b_projection: LogicalRowView
    query_key_value_z: PhysicalBlock
    query: LogicalRowView
    key: LogicalRowView
    value: LogicalRowView
    z: LogicalRowView
    norm: PhysicalBlock
    output: PhysicalBlock


@dataclass(frozen=True, slots=True)
class TextLayerBinding:
    index: int
    input_norm: PhysicalBlock
    attention: FullAttentionBinding | None
    gdn: GdnBinding | None
    post_attention_norm: PhysicalBlock
    moe: MoeBinding


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
class MtpLayerBinding:
    input_norm: PhysicalBlock
    attention: FullAttentionBinding
    post_attention_norm: PhysicalBlock
    moe: MoeBinding


@dataclass(frozen=True, slots=True)
class MtpBinding:
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
class DFlashAttentionBinding:
    query_key_value: PhysicalBlock
    query: LogicalRowView
    key: LogicalRowView
    value: LogicalRowView
    query_norm: PhysicalBlock
    key_norm: PhysicalBlock
    output: PhysicalBlock


@dataclass(frozen=True, slots=True)
class DFlashMlpBinding:
    gate_up: PhysicalBlock
    gate: LogicalRowView
    up: LogicalRowView
    down: PhysicalBlock


@dataclass(frozen=True, slots=True)
class DFlashLayerBinding:
    index: int
    input_norm: PhysicalBlock
    attention: DFlashAttentionBinding
    post_attention_norm: PhysicalBlock
    mlp: DFlashMlpBinding


@dataclass(frozen=True, slots=True)
class DFlashBinding:
    token_embedding: PhysicalBlock
    mask_embedding: LogicalRowView
    proposal_output_head: PhysicalBlock
    feature_projection: PhysicalBlock
    context_norm: PhysicalBlock
    layers: tuple[DFlashLayerBinding, ...]
    final_norm: PhysicalBlock


@dataclass(frozen=True, slots=True)
class _ExpectedTensor:
    name: str
    shape: tuple[int, ...]
    format: str | None
    layout: str


@dataclass(frozen=True, slots=True)
class _ExpectedResource:
    name: str
    encoding: str = RESOURCE_ENCODING


_ExpectedObject: TypeAlias = _ExpectedTensor | _ExpectedResource


def _tensor(name: str, shape: tuple[int, ...], format_name: str) -> _ExpectedTensor:
    layout = CONTIGUOUS if format_name in (BF16, FP32, I32) else ROW_SPLIT
    return _ExpectedTensor(name, shape, format_name, layout)


def _moe_contract(
    prefix: str,
    gate_up_format: str,
    down_format: str | None,
    cfg,
) -> tuple[_ExpectedTensor, ...]:
    return (
        _tensor(prefix + "router_shared_gate", (cfg.experts + 1, cfg.hidden), BF16),
        _tensor(prefix + "routed_gate_up", (cfg.experts * 2 * cfg.expert_intermediate, cfg.hidden), gate_up_format),
        _tensor(prefix + "routed_down", (cfg.experts * cfg.hidden, cfg.expert_intermediate), down_format),
        _tensor(prefix + "shared_gate_up", (2 * cfg.shared_intermediate, cfg.hidden), None),
        _tensor(prefix + "shared_down", (cfg.hidden, cfg.shared_intermediate), None),
    )


def _text_contract(cfg) -> tuple[_ExpectedTensor, ...]:
    tensors: list[_ExpectedTensor] = [_tensor("text/token_embedding", (cfg.vocab, cfg.hidden), None)]
    for layer in range(cfg.layers):
        prefix = f"text/layers/{layer}/"
        tensors.append(_tensor(prefix + "input_norm", (cfg.hidden,), BF16))
        if cfg.is_full(layer):
            tensors.extend(
                (
                    _tensor(
                        prefix + "attention/query_key_gate_value",
                        (cfg.attention_input_rows, cfg.hidden),
                        None,
                    ),
                    _tensor(prefix + "attention/query_norm", (cfg.head_dim,), BF16),
                    _tensor(prefix + "attention/key_norm", (cfg.head_dim,), BF16),
                    _tensor(prefix + "attention/output", (cfg.hidden, cfg.q_size), None),
                )
            )
        else:
            tensors.extend(
                (
                    _tensor(prefix + "gdn/a_log", (cfg.gdn_v_heads,), FP32),
                    _tensor(prefix + "gdn/dt_bias", (cfg.gdn_v_heads,), FP32),
                    _tensor(prefix + "gdn/convolution", (cfg.conv_width, cfg.conv_dim), BF16),
                    _tensor(prefix + "gdn/a_b_projection", (2 * cfg.gdn_v_heads, cfg.hidden), BF16),
                    _tensor(prefix + "gdn/query_key_value_z", (cfg.conv_dim + cfg.value_dim, cfg.hidden), None),
                    _tensor(prefix + "gdn/norm", (cfg.gdn_v_dim,), BF16),
                    _tensor(prefix + "gdn/output", (cfg.hidden, cfg.value_dim), None),
                )
            )
        tensors.append(_tensor(prefix + "post_attention_norm", (cfg.hidden,), BF16))
        tensors.extend(_moe_contract(prefix + "moe/", None, None, cfg))
    tensors.extend(
        (
            _tensor("text/final_norm", (cfg.hidden,), BF16),
            _tensor("text/output_head", (cfg.vocab, cfg.hidden), None),
        )
    )
    return tuple(tensors)


def _draft_contract(cfg) -> tuple[_ExpectedTensor, ...]:
    return (
        _tensor("text/draft_head", (cfg.draft_vocab, cfg.hidden), None),
        _tensor("text/draft_head_token_ids", (cfg.draft_vocab,), I32),
    )


def _mtp_contract(cfg) -> tuple[_ExpectedTensor, ...]:
    tensors = [
        _tensor("mtp/input_projection", (cfg.hidden, 2 * cfg.hidden), None),
        _tensor("mtp/embedding_norm", (cfg.hidden,), BF16),
        _tensor("mtp/hidden_norm", (cfg.hidden,), BF16),
        _tensor("mtp/layer/input_norm", (cfg.hidden,), BF16),
        _tensor("mtp/layer/attention/query_key_gate_value", (cfg.attention_input_rows, cfg.hidden), None),
        _tensor("mtp/layer/attention/query_norm", (cfg.head_dim,), BF16),
        _tensor("mtp/layer/attention/key_norm", (cfg.head_dim,), BF16),
        _tensor("mtp/layer/attention/output", (cfg.hidden, cfg.q_size), None),
        _tensor("mtp/layer/post_attention_norm", (cfg.hidden,), BF16),
    ]
    tensors.extend(_moe_contract("mtp/layer/moe/", None, None, cfg))
    tensors.append(_tensor("mtp/final_norm", (cfg.hidden,), BF16))
    return tuple(tensors)


def _vision_contract(cfg):
    return tuple(_ExpectedTensor(s.name, s.shape, s.format, s.layout) for s in dense_vision_contract(cfg))


def _dflash_contract(d) -> tuple[_ExpectedTensor, ...]:
    tensors: list[_ExpectedTensor] = [
        _tensor("dflash/feature_projection", (d.hidden, d.feature_rows), None),
        _tensor("dflash/context_norm", (d.hidden,), BF16),
    ]
    for layer in range(d.layers):
        prefix = f"dflash/layers/{layer}/"
        tensors.extend(
            (
                _tensor(prefix + "input_norm", (d.hidden,), BF16),
                _tensor(
                    prefix + "attention/query_key_value",
                    (d.q_size + 2 * d.kv_size, d.hidden),
                    None,
                ),
                _tensor(prefix + "attention/query_norm", (d.head_dim,), BF16),
                _tensor(prefix + "attention/key_norm", (d.head_dim,), BF16),
                _tensor(prefix + "attention/output", (d.hidden, d.q_size), None),
                _tensor(prefix + "post_attention_norm", (d.hidden,), BF16),
                _tensor(prefix + "mlp/gate_up", (2 * d.intermediate, d.hidden), None),
                _tensor(prefix + "mlp/down", (d.hidden, d.intermediate), None),
            )
        )
    tensors.append(_tensor("dflash/final_norm", (d.hidden,), BF16))
    return tuple(tensors)


_RESOURCE_CONTRACT = tuple(
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


def _component(name: str) -> Component:
    if name.startswith("dflash/"):
        return "dflash"
    if name.startswith("vision/"):
        return "vision"
    if name.startswith("mtp/"):
        return "mtp"
    if name in ("text/draft_head", "text/draft_head_token_ids"):
        return "draft"
    return "text"


def _validate_inventory(artifact, cfg, vision, draft):
    if artifact.identity.architecture != "qwen3_5_moe":
        raise BindingError("artifact architecture is not qwen3_5_moe")
    present = {obj.name for obj in artifact.objects}
    contract = (
        _text_contract(cfg)
        + _draft_contract(cfg)
        + (_mtp_contract(cfg) if cfg.mtp_layers else ())
        + _vision_contract(vision)
        + (_dflash_contract(draft) if draft else ())
    )
    contract = tuple(s for s in contract if s.name != "text/output_head" or s.name in present)
    expected = {s.name: s for s in contract}
    expected.update((r.name, r) for r in _RESOURCE_CONTRACT if r.name in present)
    required = {"frontend/tokenizer.json", "frontend/tokenizer_config.json", "frontend/generation_config.json"}
    if not required <= present or present != expected.keys():
        raise BindingError(
            f"reference object contract mismatch; missing={sorted(expected.keys() - present)}, extra={sorted(present - expected.keys())}"
        )
    for obj in artifact.objects:
        spec = expected[obj.name]
        if isinstance(spec, _ExpectedResource):
            if not isinstance(obj, ResourceObject) or obj.encoding != spec.encoding:
                raise BindingError(f"{obj.name}: expected a frontend resource")
            continue
        if not isinstance(obj, TensorObject) or obj.shape != spec.shape:
            raise BindingError(f"{obj.name}: tensor shape disagrees with checkpoint geometry")
        if obj.runs or obj.layout not in (CONTIGUOUS, ROW_SPLIT):
            raise BindingError(f"{obj.name}: this numerical reference supports inline BF16/groupwise weights")
        if spec.format is None:
            if obj.format not in LINEAR_FORMATS:
                raise BindingError(f"{obj.name}: unsupported reference linear format {obj.format}")
        elif obj.format != spec.format or obj.layout != spec.layout:
            raise BindingError(f"{obj.name}: control tensor format disagrees with its role")


def _row_view(
    block: PhysicalBlock,
    row_begin: int,
    row_end: int,
    views: list[LogicalRowView],
) -> LogicalRowView:
    if len(block.shape) != 2:
        raise BindingError("logical row view parent must be rank two")
    shape = (row_end - row_begin, block.shape[1])
    if row_begin < 0 or row_begin >= row_end or row_end > block.shape[0]:
        raise BindingError("logical row view is outside its physical block")
    view = LogicalRowView(block, row_begin, row_end - row_begin, shape)
    views.append(view)
    return view


def _expert_bank(
    block: PhysicalBlock,
    rows_per_expert: int,
    split_rows: int | None,
    banks: list[ExpertBank],
    experts: int,
) -> ExpertBank:
    if block.layout not in (ROW_SPLIT, CONTIGUOUS) or block.shape[0] != experts * rows_per_expert:
        raise BindingError("expert bank does not match its equal-stride geometry")
    bank = ExpertBank(block, experts, rows_per_expert, split_rows)
    banks.append(bank)
    return bank


def _attention_binding(
    prefix: str,
    blocks: dict[str, PhysicalBlock],
    row_views: list[LogicalRowView],
    cfg,
) -> FullAttentionBinding:
    parent = blocks[prefix + "query_key_gate_value"]
    return FullAttentionBinding(
        query_key_gate_value=parent,
        query=_row_view(parent, 0, cfg.q_size, row_views),
        key=_row_view(parent, cfg.q_size, cfg.q_size + cfg.kv_size, row_views),
        output_gate=_row_view(parent, cfg.q_size + cfg.kv_size, 2 * cfg.q_size + cfg.kv_size, row_views),
        value=_row_view(parent, 2 * cfg.q_size + cfg.kv_size, cfg.attention_input_rows, row_views),
        query_norm=blocks[prefix + "query_norm"],
        key_norm=blocks[prefix + "key_norm"],
        output=blocks[prefix + "output"],
    )


def _moe_binding(
    prefix: str,
    blocks: dict[str, PhysicalBlock],
    row_views: list[LogicalRowView],
    expert_banks: list[ExpertBank],
    cfg,
) -> MoeBinding:
    router_shared_gate = blocks[prefix + "router_shared_gate"]
    shared_gate_up = blocks[prefix + "shared_gate_up"]
    return MoeBinding(
        router_shared_gate=router_shared_gate,
        router=_row_view(router_shared_gate, 0, cfg.experts, row_views),
        shared_gate=_row_view(router_shared_gate, cfg.experts, cfg.experts + 1, row_views),
        routed_gate_up=_expert_bank(
            blocks[prefix + "routed_gate_up"],
            2 * cfg.expert_intermediate,
            cfg.expert_intermediate,
            expert_banks,
            cfg.experts,
        ),
        routed_down=_expert_bank(blocks[prefix + "routed_down"], cfg.hidden, None, expert_banks, cfg.experts),
        shared_gate_up=shared_gate_up,
        shared_expert_gate=_row_view(shared_gate_up, 0, cfg.shared_intermediate, row_views),
        shared_up=_row_view(shared_gate_up, cfg.shared_intermediate, 2 * cfg.shared_intermediate, row_views),
        shared_down=blocks[prefix + "shared_down"],
    )


class ArtifactBinding:
    """Complete typed target binding over one open generic artifact."""

    def __init__(self, artifact: Artifact, *, owns_artifact: bool = False):
        self.config = cfg = model_config_from_declared(artifact.geometry, layer_types=artifact.layer_types)
        self.vision_config = vision_config_from_declared(artifact.vision_geometry) if artifact.vision_geometry else None
        if self.vision_config and self.vision_config.out_hidden != cfg.hidden:
            raise BindingError("vision output width disagrees with text checkpoint")
        self.dflash_config = d = SimpleNamespace(**artifact.dflash_geometry) if artifact.dflash_geometry else None
        if d:
            d.q_size = d.query_heads * d.head_dim
            d.kv_size = d.kv_heads * d.head_dim
        _validate_inventory(artifact, cfg, self.vision_config, d)
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
            resources.get("frontend/chat_template.jinja"),
            resources["frontend/generation_config.json"],
            resources.get("frontend/preprocessor_config.json"),
            resources.get("frontend/video_preprocessor_config.json"),
        )

        row_views: list[LogicalRowView] = []
        axis_views: list[AxisView] = []
        expert_banks: list[ExpertBank] = []
        layers: list[TextLayerBinding] = []
        for layer in range(cfg.layers):
            prefix = f"text/layers/{layer}/"
            if cfg.is_full(layer):
                attention = _attention_binding(prefix + "attention/", blocks, row_views, cfg)
                gdn = None
            else:
                convolution_storage = blocks[prefix + "gdn/convolution"]
                convolution = AxisView(convolution_storage, (1, 0), (cfg.conv_dim, cfg.conv_width))
                axis_views.append(convolution)
                a_b = blocks[prefix + "gdn/a_b_projection"]
                qkvz = blocks[prefix + "gdn/query_key_value_z"]
                gdn = GdnBinding(
                    a_log=blocks[prefix + "gdn/a_log"],
                    dt_bias=blocks[prefix + "gdn/dt_bias"],
                    convolution_storage=convolution_storage,
                    convolution=convolution,
                    a_b_projection=a_b,
                    a_projection=_row_view(a_b, 0, cfg.gdn_v_heads, row_views),
                    b_projection=_row_view(a_b, cfg.gdn_v_heads, 2 * cfg.gdn_v_heads, row_views),
                    query_key_value_z=qkvz,
                    query=_row_view(qkvz, 0, cfg.key_dim, row_views),
                    key=_row_view(qkvz, cfg.key_dim, 2 * cfg.key_dim, row_views),
                    value=_row_view(qkvz, 2 * cfg.key_dim, cfg.conv_dim, row_views),
                    z=_row_view(qkvz, cfg.conv_dim, cfg.conv_dim + cfg.value_dim, row_views),
                    norm=blocks[prefix + "gdn/norm"],
                    output=blocks[prefix + "gdn/output"],
                )
                attention = None

            layers.append(
                TextLayerBinding(
                    index=layer,
                    input_norm=blocks[prefix + "input_norm"],
                    attention=attention,
                    gdn=gdn,
                    post_attention_norm=blocks[prefix + "post_attention_norm"],
                    moe=_moe_binding(prefix + "moe/", blocks, row_views, expert_banks, cfg),
                )
            )

        draft_head = DraftHeadBinding(blocks["text/draft_head"], blocks["text/draft_head_token_ids"])
        self.text = TextBinding(
            token_embedding=blocks["text/token_embedding"],
            layers=tuple(layers),
            final_norm=blocks["text/final_norm"],
            output_head=blocks.get("text/output_head", blocks["text/token_embedding"]),
            draft_head=draft_head,
        )

        self.mtp = None
        if cfg.mtp_layers:
            self.mtp = MtpBinding(
                token_embedding=self.text.token_embedding,
                full_output_head=self.text.output_head,
                optimized_proposal_head=draft_head,
                input_projection=blocks["mtp/input_projection"],
                embedding_norm=blocks["mtp/embedding_norm"],
                hidden_norm=blocks["mtp/hidden_norm"],
                layer=MtpLayerBinding(
                    input_norm=blocks["mtp/layer/input_norm"],
                    attention=_attention_binding("mtp/layer/attention/", blocks, row_views, cfg),
                    post_attention_norm=blocks["mtp/layer/post_attention_norm"],
                    moe=_moe_binding("mtp/layer/moe/", blocks, row_views, expert_banks, cfg),
                ),
                final_norm=blocks["mtp/final_norm"],
            )

        self.vision = None
        if self.vision_config:
            vision_layers: list[VisionLayerBinding] = []
            for layer in range(self.vision_config.depth):
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
            self.vision = VisionBinding(
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

        self.dflash = None
        if d:
            dflash_layers: list[DFlashLayerBinding] = []
            for layer in range(d.layers):
                prefix = f"dflash/layers/{layer}/"
                qkv = blocks[prefix + "attention/query_key_value"]
                gate_up = blocks[prefix + "mlp/gate_up"]
                dflash_layers.append(
                    DFlashLayerBinding(
                        index=layer,
                        input_norm=blocks[prefix + "input_norm"],
                        attention=DFlashAttentionBinding(
                            query_key_value=qkv,
                            query=_row_view(qkv, 0, d.q_size, row_views),
                            key=_row_view(qkv, d.q_size, d.q_size + d.kv_size, row_views),
                            value=_row_view(qkv, d.q_size + d.kv_size, d.q_size + 2 * d.kv_size, row_views),
                            query_norm=blocks[prefix + "attention/query_norm"],
                            key_norm=blocks[prefix + "attention/key_norm"],
                            output=blocks[prefix + "attention/output"],
                        ),
                        post_attention_norm=blocks[prefix + "post_attention_norm"],
                        mlp=DFlashMlpBinding(
                            gate_up=gate_up,
                            gate=_row_view(gate_up, 0, d.intermediate, row_views),
                            up=_row_view(gate_up, d.intermediate, 2 * d.intermediate, row_views),
                            down=blocks[prefix + "mlp/down"],
                        ),
                    )
                )
            self.dflash = DFlashBinding(
                token_embedding=self.text.token_embedding,
                mask_embedding=_row_view(
                    self.text.token_embedding,
                    d.mask_token,
                    d.mask_token + 1,
                    row_views,
                ),
                proposal_output_head=self.text.output_head,
                feature_projection=blocks["dflash/feature_projection"],
                context_norm=blocks["dflash/context_norm"],
                layers=tuple(dflash_layers),
                final_norm=blocks["dflash/final_norm"],
            )
        self.row_views = tuple(row_views)
        self.axis_views = tuple(axis_views)
        self.expert_banks = tuple(expert_banks)
        self._validate_draft_ids()

    @classmethod
    def open(cls, path: str | Path) -> ArtifactBinding:
        artifact = Artifact.open(path)
        try:
            return cls(artifact, owns_artifact=True)
        except BaseException:
            artifact.close()
            raise

    @classmethod
    def bind(cls, artifact: Artifact) -> ArtifactBinding:
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

    def _validate_draft_ids(self) -> None:
        block = self.text.draft_head.token_ids
        token_ids = decode_direct(self.payload(block), block.format, block.shape, device="cpu")
        if int(token_ids.min()) < 0 or int(token_ids.max()) >= self.config.token_domain:
            raise BindingError("draft token IDs are outside the checkpoint tokenizer domain")
        if torch.unique(token_ids).numel() != token_ids.numel():
            raise BindingError("draft token IDs are not unique")

    def close(self) -> None:
        if self._owns_artifact:
            self._artifact.close()

    def __enter__(self) -> ArtifactBinding:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "ArtifactBinding",
    "AxisView",
    "BindingError",
    "BoundResource",
    "DFlashAttentionBinding",
    "DFlashBinding",
    "DFlashLayerBinding",
    "DFlashMlpBinding",
    "DraftHeadBinding",
    "ExpertBank",
    "FrontendResources",
    "FullAttentionBinding",
    "GdnBinding",
    "LogicalRowView",
    "MoeBinding",
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
