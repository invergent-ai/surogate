"""Tensor-only reference operators.

The public model path uses library kernels (PyTorch SDPA and FLA) rather than
project-owned Python GPU kernels. The slow recurrence remains private as a
small-shape mathematical oracle.

An operator that depends on a dimension reads it from its own operands or takes it
as an argument, so the same code serves every size of the architecture. The family
defaults are only the fallbacks a caller with no checkpoint in hand gets.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

from .config import ATTN_SCALE, CFG, GDN_SCALE, ModelConfig


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return bf16(x.to(torch.bfloat16) @ weight.to(torch.bfloat16).t())


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    unit_offset: bool = True,
    z: torch.Tensor | None = None,
    eps: float = CFG.rms_eps,
) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    wf = weight.float() + (1.0 if unit_offset else 0.0)
    out = xf * inv * wf
    if z is not None:
        out *= F.silu(z.float())
    return bf16(out)


def l2norm(x: torch.Tensor, *, eps: float = CFG.rms_eps) -> torch.Tensor:
    xf = x.float()
    return bf16(xf * torch.rsqrt(torch.sum(xf * xf, dim=-1, keepdim=True) + eps))


def residual_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return bf16(x.float() + y.float())


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return bf16(F.silu(gate.float()) * up.float())


def sigmoid_mul(gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return bf16(torch.sigmoid(gate.float()) * x.float())


@lru_cache(maxsize=16)
def _rope_frequency(
    device_type: str, device_index: int | None, rotary_dim: int, theta: float
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    pair = torch.arange(rotary_dim // 2, device=device, dtype=torch.float32)
    return torch.pow(
        torch.tensor(theta, device=device, dtype=torch.float32),
        -2.0 * pair / rotary_dim,
    )


def apply_rope(
    x: torch.Tensor, positions: torch.Tensor, *, cfg: ModelConfig = CFG
) -> torch.Tensor:
    rotary_dim = cfg.rotary_dim
    half = rotary_dim // 2
    freq = _rope_frequency(x.device.type, x.device.index, rotary_dim, cfg.rope_theta)
    positions = positions.to(device=x.device, dtype=torch.float32)
    if positions.ndim == 1:
        angle = positions[:, None] * freq[None, :]
    elif positions.ndim == 2 and positions.shape[0] == 3:
        if positions.shape[1] != x.shape[0]:
            raise ValueError("MRoPE position length must match token count")
        axes = positions[:, :, None] * freq[None, None, :]
        angle = axes[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            end = cfg.mrope_section[axis] * 3
            angle[:, offset:end:3] = axes[axis, :, offset:end:3]
    else:
        raise ValueError("RoPE positions must have shape [T] or [3,T]")
    cos = torch.cos(angle)[:, None, :]
    sin = torch.sin(angle)[:, None, :]
    out = x.clone()
    x1 = x[:, :, :half].float()
    x2 = x[:, :, half:rotary_dim].float()
    out[:, :, :half] = bf16(x1 * cos - x2 * sin)
    out[:, :, half:rotary_dim] = bf16(x2 * cos + x1 * sin)
    return out


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    scale: float = ATTN_SCALE,
) -> torch.Tensor:
    """GQA over tensors in [T,H,D] layout."""
    qh = q.transpose(0, 1).unsqueeze(0)
    kh = k.transpose(0, 1).unsqueeze(0)
    vh = v.transpose(0, 1).unsqueeze(0)
    mask = None
    if causal and q.shape[0] != k.shape[0]:
        from torch.nn.attention.bias import causal_lower_right

        mask = causal_lower_right(q.shape[0], k.shape[0])
        causal = False
    out = F.scaled_dot_product_attention(
        qh,
        kh,
        vh,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
        enable_gqa=True,
    )
    return bf16(out.squeeze(0).transpose(0, 1))


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal depthwise conv + SiLU over channel-major [C,K] weights."""
    w = weight.contiguous().float()
    width = w.shape[1]
    sequence = torch.cat((state.float().t(), x.float()), dim=0)
    out = F.conv1d(
        sequence.t().unsqueeze(0),
        w.unsqueeze(1),
        groups=x.shape[1],
    ).squeeze(0).t()
    return bf16(F.silu(out)), sequence[-(width - 1):].t().contiguous()


def gdn_gating(
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    u = a.float() + dt_bias.float()
    softplus = torch.where(u > 20.0, u, F.softplus(u))
    return -torch.exp(a_log.float()) * softplus, torch.sigmoid(b.float())


def _value_head_map(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Which key head each value head reads, for a multi-query GDN.

    The head counts come from the operands rather than from a configuration: `q` is
    [T, key heads, key dim] and `v` is [T, value heads, value dim], and the two are
    equal on the checkpoints that do not multi-query.
    """

    key_heads = q.shape[1]
    value_heads = v.shape[1]
    if value_heads % key_heads:
        raise ValueError(
            f"{value_heads} GDN value heads do not divide over {key_heads} key heads"
        )
    return torch.arange(value_heads, device=q.device) // (value_heads // key_heads)


def gated_delta_net(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    scale: float = GDN_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FLA GDN; inputs [T,H,K]/[T,HV,V], state [1,HV,K,V]."""
    if q.shape[0] > 1 and q.device.type == "cpu":
        # CPU prefill fallback: FLA's Triton kernels are CUDA-only. Loop the
        # exact one-token recurrence below; prompt lengths in reference runs
        # are small, so the sequential form is fine as a mathematical oracle.
        outputs = []
        for t in range(q.shape[0]):
            out_t, state = gated_delta_net(
                q[t : t + 1], k[t : t + 1], v[t : t + 1],
                g[t : t + 1], beta[t : t + 1], state,
                scale=scale,
            )
            outputs.append(out_t)
        return torch.cat(outputs, dim=0), state
    if q.shape[0] == 1:
        # The recurrence is already vectorized over every value head. Keeping this
        # one-token form in PyTorch is cheap and avoids relying on FLA's recurrent
        # kernel for storage-offset-sensitive inputs.
        head_map = _value_head_map(q, v)
        value_heads, key_dim = v.shape[1], q.shape[2]
        kt = k[0].float().index_select(0, head_map)
        qt = q[0].float().index_select(0, head_map)
        next_state = state.float() * torch.exp(g[0].float()).view(1, value_heads, 1, 1)
        prediction = torch.einsum("bhkv,hk->bhv", next_state, kt)
        delta = beta[0].float().view(1, value_heads, 1) * (v[0].float() - prediction)
        next_state = next_state + kt.view(
            1, value_heads, key_dim, 1
        ) * delta.unsqueeze(-2)
        out = torch.einsum("bhkv,hk->bhv", next_state, qt) * scale
        return bf16(out.squeeze(0).unsqueeze(0)), next_state
    try:
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule as fla_chunk_gated_delta_net,
        )
    except ImportError as exc:
        raise RuntimeError(
            "the hybrid reference requires flash-linear-attention>=0.5.1 "
            "for prefill GDN"
        ) from exc
    # FLA's Triton kernels require independent aligned base pointers. Several
    # model tensors are contiguous slices with a non-zero storage offset; a
    # plain .contiguous() is allowed to return the same view and is insufficient.
    q, k, v, g, beta = (tensor.clone() for tensor in (q, k, v, g, beta))
    out, final = fla_chunk_gated_delta_net(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
        scale=scale,
        initial_state=state,
        output_final_state=True,
    )
    return bf16(out.squeeze(0)), final.float()


def _naive_gated_delta_net(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    scale: float = GDN_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small-shape oracle; never used by the reference model."""
    s = state[0].float().clone()
    out = torch.empty_like(v, dtype=torch.float32)
    head_map = _value_head_map(q, v)
    value_heads = v.shape[1]
    for token in range(q.shape[0]):
        kt = k[token].float().index_select(0, head_map)
        qt = q[token].float().index_select(0, head_map)
        s.mul_(torch.exp(g[token].float()).view(value_heads, 1, 1))
        prediction = torch.einsum("hkv,hk->hv", s, kt)
        delta = beta[token].float().unsqueeze(-1) * (v[token].float() - prediction)
        s.add_(kt.unsqueeze(-1) * delta.unsqueeze(-2))
        out[token] = torch.einsum("hkv,hk->hv", s, qt) * scale
    return bf16(out), s.unsqueeze(0)
