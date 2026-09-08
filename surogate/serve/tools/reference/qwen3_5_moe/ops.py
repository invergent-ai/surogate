"""Mathematical hybrid operations with checkpoint scales supplied by the caller."""

from __future__ import annotations
import torch
import torch.nn.functional as F
from .config import CFG
from ..qwen3_5.ops import (
    bf16,
    apply_rope,
    attention,
    gated_delta_net,
    gdn_gating,
    l2norm,
    residual_add,
    sigmoid_mul,
    silu_mul,
    _naive_gated_delta_net,
)


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    small_t: bool = False,
) -> torch.Tensor:
    """Projection with the target's prefill or Small-T weight boundary."""

    activation = x.to(torch.bfloat16)
    if small_t:
        return bf16(activation.float() @ weight.float().t())
    return bf16(activation @ weight.to(torch.bfloat16).t())


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    unit_offset: bool = True,
    z: torch.Tensor | None = None,
    eps: float = CFG.rms_eps,
) -> torch.Tensor:
    """Qwen offset RMSNorm, plus the plain gated GDN variant."""

    xf = x.float()
    inv = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    normalized = xf * inv
    if z is None:
        learned = normalized * (weight.float() + (1.0 if unit_offset else 0.0))
        return bf16(learned)

    # The source GDN oracle rounds the normalized activation before applying
    # its learned plain-RMSNorm weight, then combines the result with FP32
    # SiLU(z) before returning to the activation dtype.
    learned = bf16(normalized) * weight.to(torch.bfloat16)
    return bf16(learned.float() * F.silu(z.float()))


def causal_conv1d(x, weight, state):
    """Preserve this reference's BF16 convolution history boundary."""
    width = weight.shape[1]
    sequence = torch.cat((state.float().t(), x.float()), dim=0)
    out = F.conv1d(sequence.t().unsqueeze(0), weight.float().unsqueeze(1), groups=x.shape[1]).squeeze(0).t()
    history = sequence[-(width - 1) :].t().contiguous().to(torch.bfloat16)
    return bf16(F.silu(out)), history


__all__ = [
    "bf16",
    "apply_rope",
    "attention",
    "causal_conv1d",
    "gated_delta_net",
    "gdn_gating",
    "l2norm",
    "residual_add",
    "sigmoid_mul",
    "silu_mul",
    "_naive_gated_delta_net",
    "linear",
    "rmsnorm",
]
