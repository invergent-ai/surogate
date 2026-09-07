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

    #: Token ids the tokenizer can actually address. Distinct from `vocab`,
    #: which is the output matrix's padded row count: sampling must never be
    #: allowed to return a row that decodes to nothing. Zero means "inherit the
    #: family constant", which only a target sharing that family's tokenizer may.
    token_domain: int = 0

    #: Causal sliding-window attention. Zero is a model whose every layer sees
    #: the whole context. When set, a query at position i admits keys j with
    #: ``i - j < sliding_window`` -- exactly `sliding_window` keys including its
    #: own, which is FlashAttention's ``window_size=(sliding_window - 1, 0)`` and
    #: the convention vLLM passes for Gemma 3.
    sliding_window: int = 0

    #: Which layers attend through the window, one entry per layer, `True` for a
    #: windowed layer. The resolved schedule rather than a period, because a
    #: period cannot express what a checkpoint is allowed to state: HF's
    #: `layer_types` is an arbitrary list, and reading a missing period as 0
    #: silently windowed every layer -- including the global ones, which then
    #: also took the local rope base. Empty means the model has no window; a
    #: windowed model states one entry per layer or it is refused.
    sliding_window_schedule: tuple[bool, ...] = ()

    #: Rope base for windowed layers. Gemma 3 rotates its local layers 100x
    #: faster than its global ones, so one theta cannot describe the model.
    #: Zero means every layer uses `rope_theta`.
    sliding_rope_theta: float = 0.0

    #: Embedding scale applied after the lookup. Gemma multiplies by
    #: sqrt(hidden), downcast to the activation dtype before the multiply --
    #: doing it in fp32 and rounding after gives a different answer.
    embedding_scale: float = 0.0

    #: The denominator Gemma 3 scales its attention logits by, before the square
    #: root: `query_pre_attn_scalar ** -0.5`, which is *not* `head_dim ** -0.5`
    #: in general. The 27B has scalar 168 against head dim 128, so a target that
    #: inherited the head-dim default would be wrong by 15% and entirely silent.
    #: Zero means the model does not decouple the two and the head dim is used.
    query_pre_attn_scalar: int = 0

    def validate(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"target name {self.name!r} must be a C++ identifier")
        for field_name in ("hidden", "layers", "intermediate", "vocab"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        self.attention.validate()
        if self.linear_attention is not None:
            self.linear_attention.validate()
        if self.query_pre_attn_scalar < 0:
            raise ValueError("query_pre_attn_scalar must not be negative")
        self._validate_window()

    def _validate_window(self) -> None:
        """A windowed model states all three window fields, or none of them.

        The three are one fact in three parts -- the window itself, which layers
        are inside it, and the base those layers rotate at -- and any two without
        the third describe a model nobody meant. A half-stated window is how the
        emitted header ends up windowing layers that should see their whole
        context, which is silent: the model still answers, on the wrong keys.

        A model whose windowed layers rotate at the ordinary base states
        `sliding_rope_theta` equal to `rope_theta` rather than leaving it out, so
        that "left out" keeps meaning "not windowed".
        """

        stated = {
            "sliding_window": self.sliding_window > 0,
            "sliding_window_schedule": bool(self.sliding_window_schedule),
            "sliding_rope_theta": self.sliding_rope_theta > 0,
        }
        if any(stated.values()) and not all(stated.values()):
            missing = sorted(name for name, present in stated.items() if not present)
            present = sorted(name for name, is_set in stated.items() if is_set)
            raise ValueError(
                f"a windowed target states all three window fields or none; "
                f"{self.name} states {present} and leaves {missing} unset"
            )
        if not self.sliding_window_schedule:
            return
        if len(self.sliding_window_schedule) != self.layers:
            raise ValueError(
                f"the window schedule covers {len(self.sliding_window_schedule)} layers, "
                f"{self.name} has {self.layers}"
            )
        if not all(isinstance(flag, bool) for flag in self.sliding_window_schedule):
            raise ValueError("the window schedule must be one bool per layer")

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
        """`query_pre_attn_scalar ** -0.5` where the declaration decouples the
        two, `head_dim ** -0.5` otherwise. Gemma3-27B is the case that separates
        them: scalar 168, head dim 128, and the two differ by 15%."""
        return (self.query_pre_attn_scalar or self.attention.head_dim) ** -0.5

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
