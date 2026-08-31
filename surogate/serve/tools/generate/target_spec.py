"""One architecture, described once.

A serve target is ~1,700 lines of C++ of which ~82% is identical between any two
variants (design/unified-train-serve.md). The 18% that differs is shape
constants, tensor names, and the namespace renames that follow from the target's
name — all of which the training DSL already declares in
`surogate/dsl/models/*.py`. This module is the bridge: it reads a declaration
and produces a `TargetSpec`, which the emitters turn into target sources.

The point is not to save typing. It is that adding an architecture should be a
declaration, not an implementation, so that serving covers the same
architectures training does instead of Qwen alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AttentionSpec:
    """Full-attention shape. Head dim is explicit because it is not derivable
    from hidden/heads on every architecture — Qwen3.5 uses 256 with hidden
    2560, and Gemma/GLM/Kimi differ again."""

    query_heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int

    def validate(self) -> None:
        if self.query_heads <= 0 or self.kv_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.query_heads % self.kv_heads:
            raise ValueError(
                f"query heads ({self.query_heads}) must group evenly onto "
                f"kv heads ({self.kv_heads})"
            )
        if self.head_dim % 16:
            raise ValueError(f"head dim {self.head_dim} must be a multiple of 16")
        if not 0 < self.rotary_dim <= self.head_dim:
            raise ValueError("rotary dim must be within the head dim")


@dataclass(frozen=True)
class LinearAttentionSpec:
    """Gated-delta-net shape. `conv_kernel` is the depthwise width; the state
    width is one less, which is what the engine stores per lane."""

    key_heads: int
    key_head_dim: int
    value_heads: int
    value_head_dim: int
    conv_kernel: int

    def validate(self) -> None:
        for name, value in (
            ("key_heads", self.key_heads),
            ("key_head_dim", self.key_head_dim),
            ("value_heads", self.value_heads),
            ("value_head_dim", self.value_head_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.conv_kernel < 2:
            raise ValueError("conv kernel must be at least 2")

    @property
    def key_dim(self) -> int:
        return self.key_heads * self.key_head_dim

    @property
    def value_dim(self) -> int:
        return self.value_heads * self.value_head_dim


@dataclass(frozen=True)
class TargetSpec:
    """Everything that differs between two serve targets of the same family."""

    name: str
    hidden: int
    layers: int
    intermediate: int
    vocab: int
    rms_epsilon: float
    rope_theta: float
    attention: AttentionSpec
    linear_attention: LinearAttentionSpec | None = None
    native_context: int = 262144
    mtp_draft_tokens: int = 5
    attention_interval: int = 4
    tensor_names: dict[str, str] = field(default_factory=dict)
    #: Every learnable parameter the declaration knows about, with its HF path
    #: and adapter slices. Empty for specs written by hand; populated by
    #: ``from_dsl``. Consumed by the bindings and converter-inventory emitters,
    #: and by LoRA serving once it exists.
    params: tuple[ParamSpec, ...] = ()

    def validate(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"target name {self.name!r} must be a C++ identifier")
        for field_name in ("hidden", "layers", "intermediate", "vocab"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        self.attention.validate()
        if self.linear_attention is not None:
            self.linear_attention.validate()

    @property
    def full_attention_layers(self) -> int:
        """Mirrors qwen3_6::full_attention_layers so the emitted static_asserts
        state the same thing the header computes — if the two ever disagree the
        generated target fails to compile, which is the intent."""
        return self.layers // self.attention_interval

    @property
    def gdn_layers(self) -> int:
        return self.layers - self.full_attention_layers

    @property
    def attention_scale(self) -> float:
        return self.attention.head_dim ** -0.5

    @property
    def gdn_scale(self) -> float:
        if self.linear_attention is None:
            raise ValueError("gdn scale requires a linear-attention spec")
        return self.linear_attention.key_head_dim ** -0.5

    @property
    def query_size(self) -> int:
        return self.attention.query_heads * self.attention.head_dim

    @property
    def kv_size(self) -> int:
        return self.attention.kv_heads * self.attention.head_dim


@dataclass(frozen=True)
class LoraSlice:
    """One adapter-addressable slice of a parameter's output dimension.

    Mirrors ``surogate.dsl.specs.LoRATarget``. Fused projections carry one slice
    per logical projection — the training DSL declares ``mlp_up_weight`` as
    ``[(up, 0, 3584), (gate, 3584, 3584)]`` — which is precisely what a serving
    engine needs in order to apply an adapter trained on one logical projection
    to the right row range of a fused serve tensor. Carried through the contract
    now so that LoRA serving is later a wiring exercise rather than an
    archaeological one.
    """

    name: str
    offset: int
    size: int


@dataclass(frozen=True)
class ParamSpec:
    """One learnable parameter as the declaration sees it.

    ``dsl_name`` is the canonical (post-remap) name the training runtime binds,
    ``hf_name`` the checkpoint path it loads from — the join key every serving
    consumer needs, since the converter reads the same checkpoint.
    """

    dsl_name: str
    hf_name: str | None
    shape: tuple[str, ...]
    lora: tuple[LoraSlice, ...] = ()

    @property
    def is_lora_target(self) -> bool:
        return bool(self.lora)
