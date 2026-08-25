"""Typed binding of a generic NInfer artifact to the Qwen3.6-27B target.

The converter and this module deliberately implement the target contract on
opposite sides of the artifact boundary.  Binding resolves persistent names
once, validates the complete target inventory, and exposes only typed block
and view objects to the model hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import torch

from tools.artifact import (
    Artifact,
    ArtifactIdentity,
    ResourceObject,
    TensorObject,
    decode_direct,
)


MODEL_ID = "qwen3.6-27b"
WEIGHTS_ID = "groupwise-int"
TOKENIZER_VOCAB_SIZE = 248077

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

FULL_ATTENTION_LAYERS = tuple(range(3, 64, 4))
GDN_LAYERS = tuple(layer for layer in range(64) if layer not in FULL_ATTENTION_LAYERS)
VISION_LAYERS = tuple(range(27))

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
    query_key: PhysicalBlock
    query: LogicalRowView
    key: LogicalRowView
    gate_value: PhysicalBlock
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
    a_projection: PhysicalBlock
    b_projection: PhysicalBlock
    query_key: PhysicalBlock
    query: LogicalRowView
    key: LogicalRowView
    value_z: PhysicalBlock
    value: LogicalRowView
    norm: PhysicalBlock
    z: LogicalRowView
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
    format: str
    layout: str


@dataclass(frozen=True, slots=True)
class _ExpectedResource:
    name: str
    encoding: str = RESOURCE_ENCODING


_ExpectedObject: TypeAlias = _ExpectedTensor | _ExpectedResource


def _tensor(name: str, shape: tuple[int, ...], format_name: str) -> _ExpectedTensor:
    layout = CONTIGUOUS if format_name in (BF16, FP32, I32) else ROW_SPLIT
    return _ExpectedTensor(name, shape, format_name, layout)


def _text_contract() -> tuple[_ExpectedTensor, ...]:
    tensors: list[_ExpectedTensor] = [_tensor("text/token_embedding", (248320, 5120), Q6)]
    for layer in range(64):
        prefix = f"text/layers/{layer}/"
        tensors.append(_tensor(prefix + "input_norm", (5120,), BF16))
        if layer in FULL_ATTENTION_LAYERS:
            tensors.extend(
                (
                    _tensor(prefix + "attention/query_key", (7168, 5120), Q4),
                    _tensor(prefix + "attention/gate_value", (7168, 5120), Q5),
                    _tensor(prefix + "attention/query_norm", (256,), BF16),
                    _tensor(prefix + "attention/key_norm", (256,), BF16),
                    _tensor(prefix + "attention/output", (5120, 6144), Q5),
                )
            )
        else:
            tensors.extend(
                (
                    _tensor(prefix + "gdn/a_log", (48,), FP32),
                    _tensor(prefix + "gdn/dt_bias", (48,), FP32),
                    _tensor(prefix + "gdn/convolution", (4, 10240), BF16),
                    _tensor(prefix + "gdn/a_projection", (48, 5120), BF16),
                    _tensor(prefix + "gdn/b_projection", (48, 5120), BF16),
                    _tensor(prefix + "gdn/query_key", (4096, 5120), Q4),
                    _tensor(prefix + "gdn/value_z", (12288, 5120), Q5),
                    _tensor(prefix + "gdn/norm", (128,), BF16),
                    _tensor(prefix + "gdn/output", (5120, 6144), Q5),
                )
            )
        tensors.extend(
            (
                _tensor(prefix + "post_attention_norm", (5120,), BF16),
                _tensor(prefix + "mlp/gate_up", (34816, 5120), Q4),
                _tensor(prefix + "mlp/down", (5120, 17408), Q5),
            )
        )
    tensors.extend(
        (
            _tensor("text/final_norm", (5120,), BF16),
            _tensor("text/output_head", (248320, 5120), Q6),
        )
    )
    return tuple(tensors)


def _draft_contract() -> tuple[_ExpectedTensor, ...]:
    return (
        _tensor("text/draft_head", (131072, 5120), Q4),
        _tensor("text/draft_head_token_ids", (131072,), I32),
    )


def _mtp_contract() -> tuple[_ExpectedTensor, ...]:
    return (
        _tensor("mtp/input_projection", (5120, 10240), W8),
        _tensor("mtp/embedding_norm", (5120,), BF16),
        _tensor("mtp/hidden_norm", (5120,), BF16),
        _tensor("mtp/layer/input_norm", (5120,), BF16),
        _tensor("mtp/layer/attention/query_key_gate_value", (14336, 5120), W8),
        _tensor("mtp/layer/attention/query_norm", (256,), BF16),
        _tensor("mtp/layer/attention/key_norm", (256,), BF16),
        _tensor("mtp/layer/attention/output", (5120, 6144), W8),
        _tensor("mtp/layer/post_attention_norm", (5120,), BF16),
        _tensor("mtp/layer/mlp/gate_up", (34816, 5120), W8),
        _tensor("mtp/layer/mlp/down", (5120, 17408), W8),
        _tensor("mtp/final_norm", (5120,), BF16),
    )


def _vision_contract() -> tuple[_ExpectedTensor, ...]:
    tensors: list[_ExpectedTensor] = [
        _tensor("vision/patch_embedding", (1152, 1536), Q6),
        _tensor("vision/patch_embedding_bias", (1152,), BF16),
        _tensor("vision/position_embedding", (2304, 1152), BF16),
    ]
    for layer in VISION_LAYERS:
        prefix = f"vision/layers/{layer}/"
        tensors.extend(
            (
                _tensor(prefix + "attention/qkv", (3456, 1152), Q4),
                _tensor(prefix + "attention/qkv_bias", (3456,), BF16),
                _tensor(prefix + "attention/output", (1152, 1152), Q5),
                _tensor(prefix + "attention/output_bias", (1152,), BF16),
                _tensor(prefix + "mlp/fc1", (4304, 1152), Q4),
                _tensor(prefix + "mlp/fc1_bias", (4304,), BF16),
                _tensor(prefix + "mlp/fc2", (1152, 4304), Q5),
                _tensor(prefix + "mlp/fc2_bias", (1152,), BF16),
                _tensor(prefix + "norm1/weight", (1152,), BF16),
                _tensor(prefix + "norm1/bias", (1152,), BF16),
                _tensor(prefix + "norm2/weight", (1152,), BF16),
                _tensor(prefix + "norm2/bias", (1152,), BF16),
            )
        )
    tensors.extend(
        (
            _tensor("vision/merger/fc1", (4608, 4608), W8),
            _tensor("vision/merger/fc1_bias", (4608,), BF16),
            _tensor("vision/merger/fc2", (5120, 4608), W8),
            _tensor("vision/merger/fc2_bias", (5120,), BF16),
            _tensor("vision/merger/norm/weight", (1152,), BF16),
            _tensor("vision/merger/norm/bias", (1152,), BF16),
        )
    )
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
_TEXT_CONTRACT = _text_contract()
_DRAFT_CONTRACT = _draft_contract()
_MTP_CONTRACT = _mtp_contract()
_VISION_CONTRACT = _vision_contract()
_OBJECT_CONTRACT: tuple[_ExpectedObject, ...] = (
    _RESOURCE_CONTRACT
    + _TEXT_CONTRACT
    + _DRAFT_CONTRACT
    + _MTP_CONTRACT
    + _VISION_CONTRACT
)


def _component(name: str) -> Component:
    if name.startswith("vision/"):
        return "vision"
    if name.startswith("mtp/"):
        return "mtp"
    if name in ("text/draft_head", "text/draft_head_token_ids"):
        return "draft"
    return "text"


def _validate_inventory(artifact: Artifact) -> None:
    expected_identity = ArtifactIdentity(MODEL_ID, WEIGHTS_ID)
    if artifact.identity != expected_identity:
        raise BindingError(
            f"artifact identity is {artifact.identity!r}; expected "
            f"{expected_identity!r}"
        )
    if len(_OBJECT_CONTRACT) != 1124:
        raise RuntimeError("Qwen3.6-27B reference contract must contain 1124 objects")
    if len(artifact.objects) != len(_OBJECT_CONTRACT):
        raise BindingError(
            f"artifact has {len(artifact.objects)} objects; expected 1124"
        )
    expected_by_name = {obj.name: obj for obj in _OBJECT_CONTRACT}
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
            signature = (
                expected.name,
                expected.shape,
                expected.format,
                expected.layout,
            )
            if not isinstance(actual, TensorObject) or (
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
        _validate_inventory(artifact)
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
        for layer in range(64):
            prefix = f"text/layers/{layer}/"
            input_norm = blocks[prefix + "input_norm"]
            if layer in FULL_ATTENTION_LAYERS:
                query_key = blocks[prefix + "attention/query_key"]
                gate_value = blocks[prefix + "attention/gate_value"]
                attention = FullAttentionBinding(
                    query_key=query_key,
                    query=_row_view(query_key, 0, 6144, (6144, 5120), row_views),
                    key=_row_view(query_key, 6144, 7168, (1024, 5120), row_views),
                    gate_value=gate_value,
                    output_gate=_row_view(
                        gate_value, 0, 6144, (6144, 5120), row_views
                    ),
                    value=_row_view(
                        gate_value, 6144, 7168, (1024, 5120), row_views
                    ),
                    query_norm=blocks[prefix + "attention/query_norm"],
                    key_norm=blocks[prefix + "attention/key_norm"],
                    output=blocks[prefix + "attention/output"],
                )
                gdn = None
            else:
                query_key = blocks[prefix + "gdn/query_key"]
                value_z = blocks[prefix + "gdn/value_z"]
                convolution_storage = blocks[prefix + "gdn/convolution"]
                convolution = AxisView(
                    convolution_storage, (1, 0), (10240, 4)
                )
                axis_views.append(convolution)
                gdn = GdnBinding(
                    a_log=blocks[prefix + "gdn/a_log"],
                    dt_bias=blocks[prefix + "gdn/dt_bias"],
                    convolution_storage=convolution_storage,
                    convolution=convolution,
                    a_projection=blocks[prefix + "gdn/a_projection"],
                    b_projection=blocks[prefix + "gdn/b_projection"],
                    query_key=query_key,
                    query=_row_view(query_key, 0, 2048, (2048, 5120), row_views),
                    key=_row_view(
                        query_key, 2048, 4096, (2048, 5120), row_views
                    ),
                    value_z=value_z,
                    value=_row_view(value_z, 0, 6144, (6144, 5120), row_views),
                    norm=blocks[prefix + "gdn/norm"],
                    z=_row_view(value_z, 6144, 12288, (6144, 5120), row_views),
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
                    mlp=MlpBinding(
                        gate_up=gate_up,
                        gate=_row_view(
                            gate_up, 0, 17408, (17408, 5120), row_views
                        ),
                        up=_row_view(
                            gate_up, 17408, 34816, (17408, 5120), row_views
                        ),
                        down=blocks[prefix + "mlp/down"],
                    ),
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
        mtp_gate_up = blocks["mtp/layer/mlp/gate_up"]
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
                    query=_row_view(
                        mtp_qkgv, 0, 6144, (6144, 5120), row_views
                    ),
                    key=_row_view(
                        mtp_qkgv, 6144, 7168, (1024, 5120), row_views
                    ),
                    output_gate=_row_view(
                        mtp_qkgv, 7168, 13312, (6144, 5120), row_views
                    ),
                    value=_row_view(
                        mtp_qkgv, 13312, 14336, (1024, 5120), row_views
                    ),
                    query_norm=blocks["mtp/layer/attention/query_norm"],
                    key_norm=blocks["mtp/layer/attention/key_norm"],
                    output=blocks["mtp/layer/attention/output"],
                ),
                post_attention_norm=blocks["mtp/layer/post_attention_norm"],
                mlp=MlpBinding(
                    gate_up=mtp_gate_up,
                    gate=_row_view(
                        mtp_gate_up, 0, 17408, (17408, 5120), row_views
                    ),
                    up=_row_view(
                        mtp_gate_up, 17408, 34816, (17408, 5120), row_views
                    ),
                    down=blocks["mtp/layer/mlp/down"],
                ),
            ),
            final_norm=blocks["mtp/final_norm"],
        )

        vision_layers: list[VisionLayerBinding] = []
        for layer in VISION_LAYERS:
            prefix = f"vision/layers/{layer}/"
            vision_layers.append(
                VisionLayerBinding(
                    index=layer,
                    attention_qkv=blocks[prefix + "attention/qkv"],
                    attention_qkv_bias=blocks[prefix + "attention/qkv_bias"],
                    attention_output=blocks[prefix + "attention/output"],
                    attention_output_bias=blocks[
                        prefix + "attention/output_bias"
                    ],
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
        self.row_views = tuple(row_views)
        self.axis_views = tuple(axis_views)
        if len(self.tensors) != 1118 or len(self.row_views) != 390:
            raise RuntimeError("incomplete Qwen3.6-27B typed binding")
        if len(self.axis_views) != 48:
            raise RuntimeError("incomplete Qwen3.6-27B GDN axis binding")
        self._validate_aliases()
        self._validate_draft_ids()

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
        for layer in self.text.layers:
            if layer.gdn is None:
                continue
            view = layer.gdn.convolution
            if (
                view.block is not layer.gdn.convolution_storage
                or view.axes != (1, 0)
                or view.shape != (10240, 4)
            ):
                raise RuntimeError("GDN channel-major convolution alias is incorrect")

    def _validate_draft_ids(self) -> None:
        block = self.text.draft_head.token_ids
        token_ids = decode_direct(
            self.payload(block), block.format, block.shape, device="cpu"
        )
        if token_ids.dtype != torch.int32 or tuple(token_ids.shape) != (131072,):
            raise BindingError("draft token IDs must be I32[131072]")
        if int(token_ids.min()) < 0 or int(token_ids.max()) >= TOKENIZER_VOCAB_SIZE:
            raise BindingError("draft token IDs are outside 0..248076")
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
