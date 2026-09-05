"""What the training declaration says a serving artifact holds, and where each object comes from.

A converter used to state three things the declaration already knew: the geometry
(`LAYERS = 48`), the object list (every artifact name and shape), and the mapping —
which checkpoint tensor becomes which artifact object, and how a fused object is
assembled from several. The first two have been derived for a while; this module
derives the third, so the DSL's `hf_mapping` is the one place the checkpoint's tensor
names are spelled.

The join is the one `ServeObject.components` already promised: an object names the
declared parameters it is built from, in row order, and the declaration maps every
parameter to its checkpoint tensor. A component is a parameter name, or `param.slice`
where `slice` is one of the parameter's LoRA target names — the logical projections
inside a fused training parameter, which is how the artifact can store `gate | up`
where the trainer holds `up | gate`. The three exceptions are exactly the objects the
declaration does not describe: a repacking is an algorithm named by `transform` and
implemented here; a storage decision that cuts a fused object into typed halves is a
row cut applied after derivation (`cut_rows`); and an object with no trained
counterpart — a speculative head's token subset, a vision tower — is not derived.

Recipes speak the flat dialect (`model.layers.N...`): the shard reader folds the
VL-style `model.language_model.` nesting onto it, so one spelling reads both an
official release and a GGUF-bridged checkpoint.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Mapping, Sequence

from .inventory import FP32
from .recipe import (
    AnyOf,
    Cast,
    Concat,
    Expression,
    Reshape,
    Slice,
    SourceTensor,
    TensorRecipe,
    Transpose,
    expression_shape,
)

_LAYER_INDEX = re.compile(r"blocks\[(\d+)\]\.(.+)")
_NESTED_TEXT_PREFIX = "model.language_model."


# --------------------------------------------------------------------------------------
# Reading the declaration
# --------------------------------------------------------------------------------------


def compile_ir(architecture: str, hf_config: dict[str, Any]) -> dict[str, Any]:
    """The declaration compiled for one checkpoint config, exactly as training compiles it."""
    from surogate.dsl.py_compiler import compile_model_for_hf

    raw = compile_model_for_hf(architecture, hf_config)
    ir = json.loads(raw) if isinstance(raw, str) else raw
    if not ir.get("success"):
        raise ValueError(f"DSL compilation failed for {architecture}: {ir.get('errors')}")
    return ir


def ir_module(ir: dict[str, Any]) -> dict[str, Any]:
    modules = ir.get("modules") or []
    if not modules:
        raise ValueError("DSL IR carries no modules")
    return modules[0]


def ir_config(ir: dict[str, Any]) -> dict[str, Any]:
    return ir.get("config") or (ir_module(ir).get("config") or {})


def model_class(architecture: str) -> Any:
    """The declared model class behind an architecture string.

    The IR is a projection of the declaration and does not carry its hooks, so
    every consumer that needs one (the artifact inventory, the window schedule)
    has to come back to the class. One lookup, here, rather than one per consumer.
    """
    from surogate.dsl.decorators import _model_registry  # noqa: PLC2701 - one contract

    spec = next(
        (s for s in _model_registry.values()
         if s.hf_config and architecture in (s.hf_config.architecture, s.hf_config.model_type)),
        None,
    )
    if spec is None or not getattr(spec, "_nn_model_class", None):
        raise ValueError(f"no DSL model registered for {architecture}")
    return spec._nn_model_class  # noqa: SLF001


def symbols_for(config: dict[str, Any]) -> dict[str, int]:
    """The symbols `ServeObject.shape` entries are written against.

    Everything here is derived from the declaration; nothing is a constant that
    could disagree with it.
    """
    hidden = config["d_model"]
    # Linear-attention quantities, on the same footing as the MoE ones below: a
    # pure-attention declaration (Gemma 3, any encoder) simply has none of them,
    # and every symbol they feed resolves to 0, which the object walk then skips.
    heads_v = config.get("linear_num_value_heads", 0)
    dim_v = config.get("linear_value_head_dim", 0)
    key_dim = config.get("linear_num_key_heads", 0) * config.get("linear_key_head_dim", 0)
    value_dim = heads_v * dim_v
    conv_dim = 2 * key_dim + value_dim
    query_size = config["num_query_heads"] * config["head_size"]
    kv_size = config["num_kv_heads"] * config["head_size"]
    # MoE-only quantities: a dense declaration simply has none of them.
    experts = config.get("num_experts", 0)
    expert_ffn = config["d_ff"]
    shared = config.get("shared_expert_intermediate", 0)
    ngram, per_gram = config.get("ngram_size", 0), config.get("heads_per_ngram", 0)
    ple_heads = (ngram - 1) * per_gram if ngram else 0
    return {
        "C": hidden,
        "TwoC": 2 * hidden,
        "M": expert_ffn,
        "TwoM": 2 * expert_ffn,
        # The fused SwiGLU parameter, spelled `MUp` by the dense blocks and
        # `2M`/`TwoM` by the hybrid ones. One quantity, two names in the wild.
        "MUp": 2 * expert_ffn,
        "DraftVocab": config.get("draft_head_vocab", 0),
        "Vocab": config["vocab_size"],
        "HeadDim": config["head_size"],
        "QuerySize": query_size,
        "AttnFusedRows": 2 * query_size + 2 * kv_size,
        # No attention output gate: q, k and v stacked, nothing else.
        "QKV": query_size + 2 * kv_size,
        "AttnDim": query_size,
        "KvDim": kv_size,
        "HcCount": config.get("hc_count", 0),
        "HcWidth": config.get("hc_count", 0) * hidden,
        "HcLowRank": config.get("hc_lowrank", 0),
        # The mixing matrix a hyper-connected block learns, as wide as the streams it mixes.
        "HcMix": config.get("hc_mix", 0),
        # Kimi Delta Attention, the linear mixer GLM-5.3 runs where the other hybrids run a
        # gated delta net. Its projections are per head like theirs, but the decay is a
        # low-rank pair through a head-width bottleneck rather than a scalar per head, so the
        # geometry needs both widths.
        "KdaHeads": config.get("linear_num_heads", 0),
        "KdaHeadDim": config.get("linear_head_dim", 0),
        "KdaDim": config.get("linear_num_heads", 0) * config.get("linear_head_dim", 0),
        # q, k and v share one projection and one depthwise convolution.
        "KdaConvDim": 3 * config.get("linear_num_heads", 0) * config.get("linear_head_dim", 0),
        "KdaConvK": config.get("linear_conv_kernel_dim", 0),
        # Multi-head latent attention: the query and key/value low ranks, the query rows they
        # expand to, and the value width the output projection reads.
        "QRank": config.get("q_lora_rank", 0),
        "KVRank": config.get("kv_lora_rank", 0),
        "QDim": config["num_query_heads"] * (config.get("qk_nope_head_dim", 0)
                                             + config.get("qk_rope_head_dim", 0)),
        "VDim": config["num_query_heads"] * config.get("v_head_dim", 0),
        "KVBDim": config["num_query_heads"] * (config.get("qk_nope_head_dim", 0)
                                               + config.get("v_head_dim", 0)),
        "Hv": heads_v,
        "TwoHv": 2 * heads_v,
        "Vd": dim_v,
        "ValueDim": value_dim,
        # Three times hidden: a short-conv mixer projects to B, C and x at once.
        "ThreeC": 3 * hidden,
        # The short-conv kernel width, which is not the linear-attention one below:
        # a model may carry either mixer, and they size their convolutions apart.
        "ShortConvK": config.get("conv_kernel", 0),
        "ConvK": config.get("linear_conv_kernel_dim", 0),
        "ConvDim": conv_dim,
        "GdnFusedRows": conv_dim + value_dim,
        "E": experts,
        "RouterRows": experts + 1,
        "RoutedGateUpRows": experts * 2 * expert_ffn,
        "RoutedDownRows": experts * hidden,
        "SharedM": shared,
        "SharedGateUpRows": 2 * shared,
        "IndexerDim": config.get("indexer_head_dim", 0),
        "IndexerQueryRows": config.get("indexer_n_heads", 0) * config.get("indexer_head_dim", 0),
        "PleEmbed": config.get("ple_embed_dim", 0),
        "PleConvKernel": config.get("ple_conv_kernel_size", 0),
        "PleHeads": ple_heads,
        "PleMultipliers": 2 * ngram,
        # Vision tower, when the declaration carries one.
        "VisionHidden": config.get("vision_hidden", 0),
        "VisionIntermediate": config.get("vision_intermediate", 0),
        "VisionQkvRows": config.get("vision_qkv_rows", 0),
        "VisionPatchRows": config.get("vision_patch_rows", 0),
        "VisionPositionEmbeddings": config.get("vision_position_embeddings", 0),
        "VisionMergerHidden": config.get("vision_merger_hidden", 0),
        # DFlash scorer.
        "DflashHeadDim": config.get("dflash_head_dim", 0),
        "DflashQkvRows": config.get("dflash_qkv_rows", 0),
        "DflashAttnCols": config.get("dflash_attn_cols", 0),
        "DflashKvRows": config.get("dflash_kv_rows", 0),
        "DflashGateUpRows": config.get("dflash_gate_up_rows", 0),
        "DflashFfn": config.get("dflash_ffn", 0),
        "DflashFeatureRows": config.get("dflash_feature_rows", 0),
    }


def resolve(shape: tuple[str | int, ...], symbols: dict[str, int]) -> tuple[int, ...]:
    out = []
    for dim in shape:
        if isinstance(dim, int):
            out.append(dim)
        elif dim in symbols:
            out.append(int(symbols[dim]))
        else:
            raise KeyError(f"unknown shape symbol {dim!r}; add it to symbols_for()")
    return tuple(out)


def layer_matches(marker: str, layer: int, config: dict[str, Any]) -> bool:
    """Which layers a conditional object group lands on. `ple` follows the
    declaration's `ple_layer_ids`, which is 1-based where the engine is 0-based."""
    if marker == "ple":
        ids = config.get("ple_layer_ids") or []
        return bool(ids) and layer == ids[0] - 1
    raise ValueError(f"unknown layer-object marker {marker!r}")


def block_types(config: dict[str, Any], declared_model: Any = None) -> list[str]:
    """Which block runs at each layer, named as `_serve_blocks_` names them.

    A model may declare the derivation itself via `_serve_block_schedule_`, which
    is where it belongs: the hybrid families alternate attention against a
    linear-attention mixer, but Gemma 3 alternates *local against global
    attention*, a different axis with a different vocabulary. Without the hook
    this function would accumulate one branch per family.
    """
    declared = getattr(declared_model, "_serve_block_schedule_", None)
    if declared is not None:
        return list(declared(config))
    types = config.get("layer_types")
    if types:
        return ["attention" if t == "full_attention" else "mamba" for t in types]
    interval = config["full_attention_interval"]
    return ["attention" if (i + 1) % interval == 0 else "mamba"
            for i in range(config["n_layers"])]


# --------------------------------------------------------------------------------------
# The object list
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeclaredObject:
    """One artifact object as the declaration places it.

    `layer` is the text layer a block object sits on; `section` the `ServeSection`
    a sub-stack object belongs to (with `index` when the section repeats). `source`
    is a checkpoint tensor named directly, relative to the section's `hf_prefix`,
    for an object the training graph has no parameter for.
    """

    name: str
    shape: tuple[int, ...]
    format: str
    components: tuple[str, ...] = ()
    transform: str | None = None
    layer: int | None = None
    section: Any = None
    index: int | None = None
    source: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "format": self.format,
            "components": self.components,
            "transform": self.transform,
        }


def declared_objects(
    architecture: str,
    hf_config: dict[str, Any],
    *,
    capabilities: set[str] | None = None,
) -> list[DeclaredObject]:
    """Every tensor a serving artifact stores, from the declaration alone.

    `capabilities` names what the *target* consumes. The declaration describes
    the whole model — every Qwen3.5 checkpoint has a vision tower — but a
    serve target binds only what its C++ knows about, and the engine refuses
    to load an artifact carrying an object no binder consumes:

        artifact object was not consumed by the selected target:
        vision/patch_embedding

    So a section whose capability the target lacks is declared and not
    exported. `None` means every capability, which is what a check of the
    declaration itself wants.
    """
    return declare(architecture, hf_config).objects(capabilities=capabilities)


def inventory_for(
    architecture: str,
    hf_config: dict[str, Any],
    *,
    capabilities: set[str] | None = None,
) -> list[dict[str, Any]]:
    """`declared_objects` as plain records, for the emitters and their tests."""
    return [o.as_dict() for o in declared_objects(architecture, hf_config, capabilities=capabilities)]


# --------------------------------------------------------------------------------------
# The mapping
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LoraSlice:
    name: str
    offset: int
    size: int


@dataclass(frozen=True)
class ParamDecl:
    dsl_name: str
    shape: tuple[str, ...]
    slices: tuple[LoraSlice, ...] = ()


@dataclass(frozen=True)
class Declaration:
    """One model's declaration, compiled for one checkpoint config."""

    architecture: str
    hf_config: dict[str, Any] = field(repr=False)
    ir: dict[str, Any] = field(repr=False)
    #: Whether to name sources in the flat dialect. The shard reader canonicalises the
    #: VL-style `model.language_model.` nesting onto `model.`, so both spellings read
    #: every checkpoint; which one a target's recipes are written in is that target's
    #: settled convention, and changing it would change every name its GGUF repack plan
    #: and its preflight match against.
    flat_sources: bool = True

    @cached_property
    def module(self) -> dict[str, Any]:
        return ir_module(self.ir)

    @cached_property
    def config(self) -> dict[str, Any]:
        return ir_config(self.ir)

    @cached_property
    def hf_mapping(self) -> dict[str, Any]:
        return self.module.get("hf_mapping") or self.ir.get("hf_mapping") or {}

    @cached_property
    def model(self) -> Any:
        return model_class(self.architecture)

    @cached_property
    def symbols(self) -> dict[str, int]:
        return symbols_for(self.config)

    @cached_property
    def params(self) -> dict[str, ParamDecl]:
        forward = self.module.get("forward") or {}
        declared = {**(self.module.get("params") or {}), **(forward.get("params") or {})}
        out: dict[str, ParamDecl] = {}
        for dsl_name, entry in declared.items():
            if not isinstance(entry, dict):
                continue
            out[dsl_name] = ParamDecl(
                dsl_name=dsl_name,
                shape=tuple(str(dim) for dim in (entry.get("shape") or ())),
                slices=tuple(
                    LoraSlice(str(t.get("name", "")), int(t.get("offset", 0)), int(t.get("size", 0)))
                    for t in (entry.get("lora_targets") or [])
                ),
            )
        return out

    @cached_property
    def block_types(self) -> list[str]:
        return block_types(self.config, self.model)

    @cached_property
    def tied_output_head(self) -> bool:
        """Whether the checkpoint reuses its embedding as the output projection. The
        checkpoint states it; the declaration maps `lm_head` either way."""
        config = self.hf_config
        text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
        value = config.get("tie_word_embeddings", text.get("tie_word_embeddings"))
        return bool(value)

    # -- the object list -----------------------------------------------------------

    def objects(self, *, capabilities: set[str] | None = None) -> list[DeclaredObject]:
        model = self.model
        config = self.config
        symbols = self.symbols
        model_objects = getattr(model, "_serve_objects_", ())
        layer_objects = getattr(model, "_serve_layer_objects_", {})
        block_classes = getattr(model, "_serve_blocks_", {})
        if not block_classes:
            raise ValueError(
                f"{model.__name__} declares no _serve_blocks_; a model without serve "
                f"objects cannot have its artifact inventory derived"
            )
        out: list[DeclaredObject] = []

        def emit(name: str, obj: Any, **where: Any) -> None:
            shape = resolve(obj.shape, symbols)
            if any(dim == 0 for dim in shape):
                # The declaration describes this object but this variant has no such
                # geometry — a text-only checkpoint carries no vision tower. Declaring
                # it and resolving it away beats maintaining two object lists.
                return
            out.append(DeclaredObject(
                name=name,
                shape=shape,
                format=obj.format,
                components=tuple(obj.components),
                transform=obj.transform,
                source=getattr(obj, "source", None),
                **where,
            ))

        def supported(obj: Any) -> bool:
            return capabilities is None or obj.capability in capabilities

        model_objects = [o for o in model_objects if supported(o)]
        leading = [o for o in model_objects if o.name.endswith("token_embedding")]
        trailing = [o for o in model_objects if o not in leading]
        for obj in leading:
            emit(obj.name, obj)
        for layer, block_type in enumerate(self.block_types):
            prefix = f"text/layers/{layer}/"
            for marker, objects in layer_objects.items():
                if layer_matches(marker, layer, config):
                    for obj in objects:
                        emit(prefix + obj.name, obj, layer=layer)
            for obj in block_classes[block_type].schema.serve_objects:
                emit(prefix + obj.name, obj, layer=layer)
        for obj in trailing:
            emit(obj.name, obj)
        for section in getattr(model, "_serve_sections_", ()):
            if capabilities is not None and section.capability not in capabilities:
                continue
            count = section.repeat if isinstance(section.repeat, int) else config[section.repeat]
            for index in range(count):
                prefix = section.prefix if count == 1 else f"{section.prefix}{index}/"
                for obj in section.objects:
                    emit(prefix + obj.name, obj, section=section, index=index)
        return out

    # -- resolving one component to a checkpoint tensor -------------------------------

    def param(self, name: str, layer: int | None) -> ParamDecl:
        """The declared parameter behind a component name, at a layer or at whichever
        layer runs the block that declares it (a section replaying a block's objects)."""
        if layer is not None:
            key = f"blocks[{layer}].{name}"
            if key in self.params:
                return self.params[key]
        if name in self.params:
            return self.params[name]
        for key, decl in self.params.items():
            match = _LAYER_INDEX.match(key)
            if match and match.group(2) == name:
                return decl
        raise KeyError(f"the {self.architecture} declaration has no parameter {name!r}")

    def mapping(self, name: str, layer: int | None, hf_layer: str | None = None) -> Any:
        """The checkpoint mapping (a path, or a fuse/split record) for one parameter.

        Per-layer params arrive as `blocks[7].x`; the mapping may carry that exact key
        (the hybrid expansion emits physical indices) or the bare name with a `{layer}`
        placeholder. `hf_layer` re-roots the bare template under another layer prefix:
        a draft head's decoder layer is a text block stored under `mtp.layers.0`.
        """
        if hf_layer is not None:
            template = self.hf_mapping.get(name)
            if template is None:
                raise KeyError(f"no layer template maps {name!r}; cannot re-root it under {hf_layer}")
            return _reroot(template, hf_layer)
        if layer is not None:
            direct = self.hf_mapping.get(f"blocks[{layer}].{name}")
            if direct is not None:
                return direct
            template = self.hf_mapping.get(name)
            if template is not None:
                return _substitute(template, "{layer}", str(layer))
        direct = self.hf_mapping.get(name)
        if direct is None:
            raise KeyError(f"the {self.architecture} declaration maps no checkpoint tensor to {name!r}")
        return direct

    def dims(self, decl: ParamDecl) -> tuple[int, ...]:
        return tuple(_dim(text, self.symbols, self.config) for text in decl.shape)

    def source_name(self, name: str) -> str:
        return flat_name(name) if self.flat_sources else name

    def _stacked_experts(self, name: str, mapping: dict[str, Any],
                         shape: tuple[int, ...]) -> Expression:
        """A mixture's experts, stacked into the expert-major parameter the graph holds.

        HuggingFace stores one tensor per expert per projection; the training parameter is
        `[experts, rows, cols]`. The declaration says so with a `stack_experts` record naming
        the pattern, and the artifact's `flatten_experts` transform then writes those numbers as
        rows -- so what this has to produce is exactly the batched parameter, not the rows.

        `fuse_gate_up` means the pattern names the gate projection and each expert also has an
        `up_proj` beside it, the two making one expert's `[2M, C]`. They concatenate on the
        expert's own output axis, which is the order the fused parameter has; concatenating the
        experts on that axis instead would interleave them.
        """
        if len(shape) != 3:
            raise ValueError(
                f"{name!r} stacks per-expert tensors, so its parameter is "
                f"[experts, rows, cols]; the declaration says {shape}"
            )
        experts, rows, cols = shape
        pattern = str(mapping["pattern"])
        up_pattern = str(mapping.get("up_pattern") or "")
        fuse_gate_up = bool(mapping.get("fuse_gate_up", False))
        if fuse_gate_up and not up_pattern:
            # The conventional spelling: the gate's own name with `gate` replaced.
            up_pattern = pattern.replace("gate_proj", "up_proj")
            if up_pattern == pattern:
                raise ValueError(
                    f"{name!r} asks to fuse gate and up, and its pattern {pattern!r} does not "
                    f"name a gate projection to find the up beside"
                )
        halves = 2 if up_pattern else 1
        if rows % halves != 0:
            raise ValueError(
                f"{name!r} fuses {halves} projections per expert, which does not divide its "
                f"{rows} rows"
            )
        per_half = rows // halves
        per_expert: list[Expression] = []
        for expert in range(experts):
            def tensor(source: str) -> Expression:
                return SourceTensor(self.source_name(source.format(expert=expert)),
                                    (per_half, cols))
            one = tensor(pattern) if halves == 1 else Concat(
                (tensor(pattern), tensor(up_pattern)), 0)
            per_expert.append(Reshape(one, (1, rows, cols)))
        stacked = Concat(tuple(per_expert), 0)

        # A GGUF holds a layer's experts in one tensor per projection, already expert-major,
        # and the bridge keeps them that way -- the same numbers in the same order as the
        # per-expert files above, under a name with no expert in it. That is a source-format
        # convention rather than architecture, so both spellings are offered and the resolver
        # takes whichever the checkpoint at hand has.
        def whole(source: str) -> Expression:
            return SourceTensor(self.source_name(source.replace("{expert}.", "")),
                                (experts, per_half, cols))
        if "{expert}." not in pattern:
            return stacked
        # Concatenated on the expert's *output* axis, which is the order the fused parameter
        # has; concatenating on the expert axis would interleave the two projections.
        together = whole(pattern) if halves == 1 else Concat(
            (whole(pattern), whole(up_pattern)), 1)
        return AnyOf((stacked, together))

    def component(self, component: str, *, layer: int | None, hf_layer: str | None = None,
                  shape: tuple[int, ...] | None = None) -> Expression:
        """The expression producing one component: a whole parameter's checkpoint tensor,
        a fused parameter reassembled from its sources, or one named slice of it.

        `shape` is the declared shape of an object that has this component and no other,
        used when the declaration maps a checkpoint tensor under a name its graph does
        not register as a parameter — Gemma 3 reads its QK norms inside the attention
        module rather than tracing them. The mapping still says where the tensor lives;
        the object says how big it is. Anything more than a whole-tensor pass-through
        still needs the parameter, because only it carries the fused parts.
        """
        name, _, slice_name = component.partition(".")
        try:
            decl = self.param(name, layer)
        except KeyError:
            untraced = None if slice_name or shape is None else self.mapping(name, layer, hf_layer)
            if not isinstance(untraced, str):
                raise
            return SourceTensor(self.source_name(untraced), shape)
        mapping = self.mapping(name, layer, hf_layer)
        shape = self.dims(decl)
        if isinstance(mapping, str):
            if slice_name:
                raise ValueError(
                    f"{component}: {name!r} maps to one checkpoint tensor and has no slices to name"
                )
            return SourceTensor(self.source_name(mapping), shape)
        if isinstance(mapping, dict) and mapping.get("type") == "stack_experts":
            if slice_name:
                raise ValueError(
                    f"{component}: {name!r} stacks per-expert tensors and has no slices to name"
                )
            return self._stacked_experts(name, mapping, shape)
        if not isinstance(mapping, dict) or mapping.get("type") != "fuse":
            raise NotImplementedError(
                f"{component}: {name!r} is mapped by a {mapping.get('type') if isinstance(mapping, dict) else type(mapping).__name__} "
                f"record, which recipe derivation does not express yet"
            )
        sources = list(mapping["sources"])
        dim = int(mapping.get("dim", 0))
        slices = decl.slices
        if len(slices) != len(sources):
            raise ValueError(
                f"{name!r} fuses {len(sources)} tensors but declares {len(slices)} LoRA slices; "
                f"the slices are what name the fused parts, so they must correspond"
            )
        expected = 0
        for lora, source in zip(slices, sources):
            if lora.offset != expected:
                raise ValueError(f"{name!r}: slice {lora.name!r} at {lora.offset} is not contiguous with the fuse order")
            expected += lora.size
        if expected != shape[dim]:
            raise ValueError(f"{name!r}: slices span {expected} rows, the parameter has {shape[dim]}")

        def part(lora: LoraSlice, source: str) -> SourceTensor:
            part_shape = list(shape)
            part_shape[dim] = lora.size
            return SourceTensor(self.source_name(source), tuple(part_shape))

        if slice_name:
            for lora, source in zip(slices, sources):
                if lora.name == slice_name:
                    return part(lora, source)
            raise KeyError(f"{name!r} has no slice {slice_name!r}; it has {[s.name for s in slices]}")
        return Concat(tuple(part(lora, source) for lora, source in zip(slices, sources)), dim)


def text_config(
    architecture: str,
    model_type: str,
    *,
    layers: int,
    hidden: int,
    intermediate: int,
    vocab: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    **extra: Any,
) -> dict[str, Any]:
    """The `config.json` a text checkpoint of these dimensions would carry.

    The inverse of a converter's `geometry_from_config`, and the reason a caller with
    a registered geometry and no checkpoint in hand can still derive recipes: the
    declaration is compiled against a config either way. Only the dimensions the
    object list is a function of are stated; `extra` carries whatever else a
    particular family's declaration reads.
    """
    return {
        "architectures": [architecture],
        "model_type": model_type,
        "num_hidden_layers": layers,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "num_attention_heads": query_heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        **extra,
    }


def _sources_of(obj: DeclaredObject) -> tuple[tuple[str, tuple[str | int, ...] | None], ...]:
    """One object's checkpoint tensors, each with its own shape or `None` for the whole
    object. One name is the object itself; several are row-concatenated in the order
    given, and each then states its shape because the object declares only their total."""
    source = obj.source
    if isinstance(source, str):
        return ((source, None),)
    if all(isinstance(entry, str) for entry in source):
        raise ValueError(
            f"{obj.name}: an object built from several checkpoint tensors states each one's "
            f"shape, as (name, shape) pairs — only their total rows are declared"
        )
    return tuple((name, shape) for name, shape in source)


def declare(architecture: str, hf_config: dict[str, Any], *,
            flat_sources: bool = True) -> Declaration:
    return Declaration(
        architecture=architecture,
        hf_config=hf_config,
        ir=compile_ir(architecture, hf_config),
        flat_sources=flat_sources,
    )


def flat_name(name: str) -> str:
    """The flat spelling of a checkpoint tensor. Official Qwen3.5 releases nest the text
    tower under `model.language_model.`; the shard reader folds that onto `model.`, and
    recipes are written in the folded form so one spelling reads every source."""
    if name.startswith(_NESTED_TEXT_PREFIX):
        return "model." + name[len(_NESTED_TEXT_PREFIX):]
    return name


def _substitute(mapping: Any, placeholder: str, value: str) -> Any:
    if isinstance(mapping, str):
        return mapping.replace(placeholder, value)
    if isinstance(mapping, dict):
        return {k: _substitute(v, placeholder, value) for k, v in mapping.items()}
    if isinstance(mapping, list):
        return [_substitute(v, placeholder, value) for v in mapping]
    return mapping


def _reroot(mapping: Any, hf_layer: str) -> Any:
    """`model.layers.{layer}.self_attn.q_proj.weight` under `mtp.layers.0` is
    `mtp.layers.0.self_attn.q_proj.weight`: the part after the layer placeholder is the
    block's own naming, and it is the same block."""
    if isinstance(mapping, str):
        marker = "{layer}."
        at = mapping.find(marker)
        if at < 0:
            raise ValueError(f"{mapping!r} carries no {{layer}} placeholder to re-root")
        return f"{hf_layer}.{mapping[at + len(marker):]}"
    if isinstance(mapping, dict):
        return {k: _reroot(v, hf_layer) if k == "sources" else v for k, v in mapping.items()}
    if isinstance(mapping, list):
        return [_reroot(v, hf_layer) for v in mapping]
    return mapping


def _dim(text: str, symbols: Mapping[str, int], config: Mapping[str, Any]) -> int:
    """One parameter dimension as the IR spells it: a literal, a shape symbol, or an
    arithmetic expression over the runtime config (`64 // 2`)."""
    if text.isdigit():
        return int(text)
    if text in symbols:
        return int(symbols[text])
    names = {k: v for k, v in config.items() if isinstance(v, int) and not isinstance(v, bool)}
    names.update(symbols)
    try:
        value = eval(text, {"__builtins__": {}}, names)  # noqa: S307 - the IR's own arithmetic
    except Exception as error:  # noqa: BLE001
        raise KeyError(f"cannot resolve parameter dimension {text!r}") from error
    if not isinstance(value, int):
        raise KeyError(f"parameter dimension {text!r} resolves to {value!r}, not an int")
    return value


# --------------------------------------------------------------------------------------
# Repackings the declaration names and this module implements
# --------------------------------------------------------------------------------------

Transform = Callable[[Sequence[Expression], DeclaredObject, Declaration], Expression]


def _concat_rows(parts: Sequence[Expression], obj: DeclaredObject, decl: Declaration) -> Expression:
    if len(parts) == 1:
        return parts[0]
    return Concat(tuple(parts), 0)


def _split_interleaved_query_gate(
    parts: Sequence[Expression], obj: DeclaredObject, decl: Declaration
) -> Expression:
    """q | k | gate | v from a query projection that interleaves query and gate per head."""
    query_proj, key, value = parts
    head_dim = decl.symbols["HeadDim"]
    rows, hidden = expression_shape(query_proj)
    heads = rows // (2 * head_dim)
    per_head = Reshape(query_proj, (heads, 2 * head_dim, hidden))

    def half(begin: int) -> Expression:
        return Reshape(Slice(per_head, 1, begin, begin + head_dim), (heads * head_dim, hidden))

    return Concat((half(0), key, half(head_dim), value), 0)


def _transpose_taps(parts: Sequence[Expression], obj: DeclaredObject, decl: Declaration) -> Expression:
    """A depthwise convolution stored `[channels, 1, taps]` served as `[taps, channels]`."""
    (convolution,) = parts
    channels, _, taps = expression_shape(convolution)
    return Transpose(Reshape(Slice(convolution, 1, 0, 1), (channels, taps)), (1, 0))


def _flatten_experts(parts: Sequence[Expression], obj: DeclaredObject, decl: Declaration) -> Expression:
    """An expert-major parameter `[experts, rows, cols]` as the rows the artifact stores.

    The training graph holds every expert of a layer in one batched parameter; the
    serving artifact stores the same numbers as a plain row-major matrix, so
    `stored_row(e, r) = e * rows + r` is a contiguous reshape rather than a permutation.

    A checkpoint that fuses gate and up into one expert-major tensor and a GGUF of the
    same model that keeps them as two stacked tensors describe identical rows in
    identical order. That is a source-format convention like the `model.language_model.`
    folding, not architecture, so the object accepts both spellings and the resolver
    takes whichever one the checkpoint at hand provides.
    """
    (batched,) = parts
    shape = expression_shape(batched)
    if len(shape) != 3:
        raise ValueError(
            f"{obj.name}: flatten_experts wants an expert-major [experts, rows, cols] "
            f"source, and this one is {shape}"
        )
    experts, rows, cols = shape
    fused = Reshape(batched, obj.shape)
    if not isinstance(batched, SourceTensor) or _FUSED_GATE_UP not in batched.name:
        return fused
    stacked = tuple(
        SourceTensor(batched.name.replace(_FUSED_GATE_UP, half), (experts, rows // 2, cols))
        for half in ("gate_proj", "up_proj")
    )
    # Concatenated on the expert's *output* axis, which is what reproduces the fused
    # tensor's row order; concatenating on the expert axis would interleave them.
    return AnyOf((fused, Reshape(Concat(stacked, 1), obj.shape)))


#: The fused expert projection's name, and the token that gives away its two halves.
_FUSED_GATE_UP = "gate_up_proj"


#: Transforms whose work happens at load or in the kernel: the recipe passes the
#: tensor through and the name only records that the served form differs.
TRANSFORMS: dict[str | None, Transform] = {
    None: _concat_rows,
    "split_interleaved_query_gate": _split_interleaved_query_gate,
    "transpose_taps": _transpose_taps,
    "flatten_experts": _flatten_experts,
    "log_negate": _concat_rows,
    "unfold_unit_offset": _concat_rows,
}

#: Components whose checkpoint tensor is another component's when the checkpoint ties
#: them. The declaration maps both; the checkpoint says whether they are one tensor.
TIED_COMPONENTS = {"lm_head": "embedding"}


def derive_recipes(
    declaration: Declaration,
    *,
    capabilities: set[str] | None = None,
    tied_output_head: bool | None = None,
    transforms: Mapping[str | None, Transform] | None = None,
) -> tuple[TensorRecipe, ...]:
    """Where every derivable artifact object comes from, in declaration order.

    Objects with neither components nor a direct source are not derived: they are
    the ones the training graph does not describe, and their recipes are policy the
    converter states itself. Every derived expression is checked against the
    declared object shape, which is the proof that the declaration's row order and
    the checkpoint's tensors agree.
    """
    transforms = TRANSFORMS if transforms is None else transforms
    tied = declaration.tied_output_head if tied_output_head is None else tied_output_head
    recipes: list[TensorRecipe] = []
    for obj in declaration.objects(capabilities=capabilities):
        if obj.source is not None:
            prefix = getattr(obj.section, "hf_prefix", "") if obj.section is not None else ""
            if obj.index is not None:
                prefix = prefix.replace("{index}", str(obj.index))
            parts = [
                SourceTensor(
                    declaration.source_name(prefix + name),
                    obj.shape if shape is None else resolve(shape, declaration.symbols),
                )
                for name, shape in _sources_of(obj)
            ]
            expression: Expression = _concat_rows(parts, obj, declaration)
        elif obj.components:
            hf_layer = getattr(obj.section, "hf_layer", None) if obj.section is not None else None
            if hf_layer is not None and obj.index is not None:
                hf_layer = hf_layer.replace("{index}", str(obj.index))
            # An object built from one component is that component, so its declared
            # shape can stand in for a parameter the graph does not trace.
            only = obj.shape if len(obj.components) == 1 else None
            parts = []
            for component in obj.components:
                if tied and component in TIED_COMPONENTS:
                    tensor = declaration.component(TIED_COMPONENTS[component], layer=obj.layer, hf_layer=hf_layer)
                    own = declaration.dims(declaration.param(component, obj.layer))
                    if expression_shape(tensor) != own:
                        raise ValueError(f"{obj.name}: {component} is tied to a tensor of another shape")
                    parts.append(tensor)
                else:
                    parts.append(declaration.component(component, layer=obj.layer,
                                                       hf_layer=hf_layer, shape=only))
            if obj.transform not in transforms:
                raise KeyError(f"{obj.name}: transform {obj.transform!r} has no implementation")
            expression = transforms[obj.transform](parts, obj, declaration)
        else:
            continue
        if obj.format == "fp32":
            expression = Cast(expression, FP32)
        actual = expression_shape(expression)
        if actual != obj.shape:
            raise ValueError(
                f"{obj.name}: the declaration says {obj.shape} but its components assemble to {actual}"
            )
        recipes.append(TensorRecipe(obj.name, expression))
    return tuple(recipes)


# --------------------------------------------------------------------------------------
# Storage decisions applied after derivation
# --------------------------------------------------------------------------------------


def slice_rows(expression: Expression, begin: int, end: int) -> Expression:
    """Rows `[begin, end)` of an expression, folded through a row concatenation so a cut on
    a part boundary names the parts rather than slicing their concatenation."""
    rows = expression_shape(expression)[0]
    if begin == 0 and end == rows:
        return expression
    if isinstance(expression, Concat) and expression.axis == 0:
        kept: list[Expression] = []
        at = 0
        for part in expression.sources:
            size = expression_shape(part)[0]
            lo, hi = max(begin, at), min(end, at + size)
            if lo < hi:
                kept.append(slice_rows(part, lo - at, hi - at))
            at += size
        return kept[0] if len(kept) == 1 else Concat(tuple(kept), 0)
    return Slice(expression, 0, begin, end)


def cut_rows(
    recipes: Sequence[TensorRecipe],
    cuts: Mapping[str, Sequence[tuple[str, int]]],
    *,
    prefix: str = "",
) -> tuple[TensorRecipe, ...]:
    """`recipes` with each object named in `cuts` replaced by its row parts, in order:
    `{"attention/query_key_gate_value": (("attention/query_key", rows), ("attention/gate_value", rows))}`.
    Keys are matched by suffix under `prefix`, so one cut applies to every text layer and
    leaves a draft head's replay of the same block alone."""
    out: list[TensorRecipe] = []
    for recipe in recipes:
        name = recipe.object_name
        suffix = next(
            (s for s in cuts if name.startswith(prefix) and name.endswith(s)), None
        )
        if suffix is None:
            out.append(recipe)
            continue
        stem = name[: len(name) - len(suffix)]
        total = expression_shape(recipe.expression)[0]
        at = 0
        for part, rows in cuts[suffix]:
            out.append(TensorRecipe(stem + part, slice_rows(recipe.expression, at, at + rows)))
            at += rows
        if at != total:
            raise ValueError(f"{name}: the cut covers {at} rows of {total}")
    return tuple(out)
