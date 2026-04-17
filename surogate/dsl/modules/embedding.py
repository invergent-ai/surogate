"""Embedding / LM-head modules."""

from __future__ import annotations

from typing import Any

from ..dim import B, T
from ..nn import Module, Proxy, Tracer
from ..specs import ActivationScope


class Embedding(Module):
    """Embedding lookup table."""

    _hf_mapping_defaults_ = {
        "weight": "model.embed_tokens.weight",
    }

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (token_ids,) = args

        # Keep token embeddings full-precision in QLoRA flows.
        w = tracer.register_param(
            "weight",
            ("vocab_size", "d_model"),
            quantizable=False,
        )

        out_slot = tracer.register_activation(
            "out",
            ("B", "T", "d_model"),
            scope=ActivationScope.GLOBAL,
            description="Embedded input",
        )

        out = g.embedding(token_ids.ref, w, out_name=out_slot)
        return Proxy(out_slot, out)


class ScaledEmbedding(Module):
    """Embedding lookup with output scaling.

    Used by Gemma-family models where embed_tokens output is multiplied by
    a constant (typically sqrt(hidden_size)) before being fed into the
    decoder stack.

    Args:
        vocab_size: Vocabulary size.
        d_model: Embedding dimension.
        embed_scale: Explicit scale factor. If ``None`` (default), uses
            ``sqrt(d_model)``.
    """

    _hf_mapping_defaults_ = {
        "weight": "model.embed_tokens.weight",
    }

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        embed_scale: float | None = None,
        dim_name: str = "d_model",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed_scale = embed_scale if embed_scale is not None else float(d_model) ** 0.5
        self._dim_name = dim_name

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (token_ids,) = args
        dim = self._dim_name

        w = tracer.register_param(
            "weight",
            ("vocab_size", dim),
            quantizable=False,
        )

        out_slot = tracer.register_activation(
            "out",
            ("B", "T", dim),
            scope=ActivationScope.GLOBAL,
            description="Scaled embedded input",
        )

        raw = g.embedding(token_ids.ref, w)
        scaled = g.scale(raw, factor=self.embed_scale)
        # Bind the scale output to the registered activation slot name
        out = g.view(scaled, shape=[B, T, dim], out_name=out_slot)
        return Proxy(out_slot, out)


class LMHead(Module):
    """Fused LM head projection + cross-entropy loss."""

    _hf_mapping_defaults_ = {
        "weight": "lm_head.weight",
    }

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        softcap: float | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.softcap = softcap

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        x, targets = args

        # Keep LM head full-precision in QLoRA flows.
        w = tracer.register_param(
            "weight",
            ("vocab_size", "d_model"),
            quantizable=False,
        )

        loss_slot = tracer.register_activation(
            "loss",
            ("B * T",),
            dtype="fp32",
            scope=ActivationScope.GLOBAL,
            description="Cross-entropy loss per token",
        )

        x_flat = g.view(
            x.ref,
            shape=["B * T", "d_model"],
            out_name=tracer.prefixed("x_flat"),
        )
        loss = g.fused_lm_head_loss(
            x_flat,
            w,
            targets.ref,
            compute_accuracy=True,
            softcap=self.softcap,
            out_name=loss_slot,
        )
        return Proxy(loss_slot, loss)


# ============================================================================
# Canonical name remaps — map auto-prefixed nn names to C++ runtime names
# ============================================================================

# Dense transformer block: attn_norm / self_attn / mlp_norm / mlp
