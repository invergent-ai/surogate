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
    tensor_names: dict[str, str] = field(default_factory=dict)

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
    def query_size(self) -> int:
        return self.attention.query_heads * self.attention.head_dim

    @property
    def kv_size(self) -> int:
        return self.attention.kv_heads * self.attention.head_dim
