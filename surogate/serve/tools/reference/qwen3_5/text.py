"""Explicit text-core layer schedule for the hybrid GDN/attention decoder."""

from __future__ import annotations

import torch

from surogate.serve.tools.reference.common.tap import NullTap

from .ops import (
    apply_rope,
    causal_conv1d,
    gated_delta_net,
    gdn_gating,
    l2norm,
    linear,
    residual_add,
    rmsnorm,
    sigmoid_mul,
    silu_mul,
)


def attention_mixer(model, layer, x, positions, start, tap, context) -> torch.Tensor:
    cfg = model.config
    layer_weights = model.binding.text.layers[layer]
    attention_weights = layer_weights.attention
    h = rmsnorm(x, model.weight(layer_weights.input_norm), eps=cfg.rms_eps)
    qk = linear(h, model.block_weight(attention_weights.query_key))
    gatev = linear(h, model.block_weight(attention_weights.gate_value))
    q = qk[:, :cfg.q_size].reshape(-1, cfg.q_heads, cfg.head_dim)
    k = qk[:, cfg.q_size:].reshape(-1, cfg.kv_heads, cfg.head_dim)
    gate = gatev[:, :cfg.q_size].reshape(-1, cfg.q_heads, cfg.head_dim)
    v = gatev[:, cfg.q_size:].reshape(-1, cfg.kv_heads, cfg.head_dim)
    q = apply_rope(
        rmsnorm(q, model.weight(attention_weights.query_norm), eps=cfg.rms_eps),
        positions,
        cfg=cfg,
    )
    k = apply_rope(
        rmsnorm(k, model.weight(attention_weights.key_norm), eps=cfg.rms_eps),
        positions,
        cfg=cfg,
    )
    if tap.level == "op":
        for name, value in (("q", q), ("k", k), ("v", v), ("gate", gate)):
            model._tap(tap, f"layer_{layer:02d}/op/{name}", value, **context)
    attended = model._gqa(q, k, v, cfg.full_index(layer), start)
    out = linear(
        sigmoid_mul(gate, attended).reshape(-1, cfg.q_size),
        model.weight(attention_weights.output),
    )
    return residual_add(x, out)


def gdn_mixer(model, layer, x, tap, context) -> torch.Tensor:
    cfg = model.config
    _, state = model._ready()
    layer_weights = model.binding.text.layers[layer]
    gdn_weights = layer_weights.gdn
    index = cfg.gdn_index(layer)
    h = rmsnorm(x, model.weight(layer_weights.input_norm), eps=cfg.rms_eps)
    qk = linear(h, model.block_weight(gdn_weights.query_key))
    value = linear(h, model.block_weight(gdn_weights.value))
    qkv = torch.cat((qk[:, :cfg.key_dim], qk[:, cfg.key_dim:], value), dim=-1)
    a = linear(h, model.weight(gdn_weights.a_projection))
    b = linear(h, model.weight(gdn_weights.b_projection))
    qkv, state.conv[index] = causal_conv1d(
        qkv,
        model.weight(gdn_weights.convolution),
        state.conv[index],
    )
    g, beta = gdn_gating(
        a,
        b,
        model.weight(gdn_weights.a_log),
        model.weight(gdn_weights.dt_bias),
    )
    q = l2norm(
        qkv[:, :cfg.key_dim].reshape(-1, cfg.gdn_k_heads, cfg.gdn_k_dim),
        eps=cfg.rms_eps,
    )
    k = l2norm(
        qkv[:, cfg.key_dim:2 * cfg.key_dim].reshape(
            -1, cfg.gdn_k_heads, cfg.gdn_k_dim
        ),
        eps=cfg.rms_eps,
    )
    value = qkv[:, 2 * cfg.key_dim:].reshape(-1, cfg.gdn_v_heads, cfg.gdn_v_dim)
    out, state.ssm[index] = gated_delta_net(
        q, k, value, g, beta, state.ssm[index], scale=cfg.gdn_scale
    )
    z = linear(h, model.weight(gdn_weights.z)).reshape(
        -1, cfg.gdn_v_heads, cfg.gdn_v_dim
    )
    if tap.level == "op":
        for name, tensor in (("conv", qkv), ("g", g), ("beta", beta), ("gdn", out)):
            model._tap(tap, f"layer_{layer:02d}/op/{name}", tensor, **context)
    out = rmsnorm(
        out,
        model.weight(gdn_weights.norm),
        unit_offset=False,
        z=z,
        eps=cfg.rms_eps,
    )
    return residual_add(
        x,
        linear(
            out.reshape(-1, cfg.value_dim),
            model.weight(gdn_weights.output),
        ),
    )


def mlp(model, layer, x) -> torch.Tensor:
    cfg = model.config
    layer_weights = model.binding.text.layers[layer]
    h = rmsnorm(x, model.weight(layer_weights.post_attention_norm), eps=cfg.rms_eps)
    gate_up = linear(h, model.block_weight(layer_weights.mlp.gate_up))
    gate, up = gate_up.split(cfg.intermediate, dim=-1)
    return residual_add(
        x,
        linear(silu_mul(gate, up), model.weight(layer_weights.mlp.down)),
    )


def run(
    model,
    x: torch.Tensor,
    positions: torch.Tensor,
    start: int,
    *,
    phase: str,
    step: int,
    chunk: int,
    tap=None,
) -> torch.Tensor:
    cfg = model.config
    tap = tap or NullTap()
    context = dict(phase=phase, step=step, chunk=chunk, position=start)
    for layer in range(cfg.layers):
        x = (
            attention_mixer(model, layer, x, positions, start, tap, context)
            if cfg.is_full(layer)
            else gdn_mixer(model, layer, x, tap, context)
        )
        model._tap(tap, f"layer_{layer:02d}/mixer", x, **context)
        x = mlp(model, layer, x)
        model._tap(tap, f"layer_{layer:02d}/mlp", x, **context)
    return x
