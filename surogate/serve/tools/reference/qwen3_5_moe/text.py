"""Checkpoint-driven Text schedule for hybrid MoE."""

from __future__ import annotations

import torch

from surogate.serve.tools.reference.common.tap import NullTap

from .moe import forward as sparse_moe
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
)


def attention_mixer(
    model,
    layer,
    x,
    positions,
    start,
    tap,
    context,
    *,
    small_t,
) -> torch.Tensor:
    layer_weights = model.binding.text.layers[layer]
    attention_weights = layer_weights.attention
    h = rmsnorm(x, model.weight(layer_weights.input_norm), eps=model.config.rms_eps)
    projected = linear(
        h,
        model.block_weight(
            attention_weights.query_key_gate_value,
            small_t=small_t,
        ),
        small_t=small_t,
    )
    q1 = model.config.q_size
    k1 = q1 + model.config.kv_size
    gate1 = k1 + model.config.q_size
    q = projected[:, :q1].reshape(-1, model.config.q_heads, model.config.head_dim)
    k = projected[:, q1:k1].reshape(-1, model.config.kv_heads, model.config.head_dim)
    gate = projected[:, k1:gate1].reshape(-1, model.config.q_heads, model.config.head_dim)
    value = projected[:, gate1:].reshape(-1, model.config.kv_heads, model.config.head_dim)
    q = apply_rope(
        rmsnorm(q, model.weight(attention_weights.query_norm), eps=model.config.rms_eps), positions, cfg=model.config
    )
    k = apply_rope(
        rmsnorm(k, model.weight(attention_weights.key_norm), eps=model.config.rms_eps), positions, cfg=model.config
    )
    if tap.level == "op":
        for name, tensor in (("q", q), ("k", k), ("v", value), ("gate", gate)):
            model._tap(tap, f"layer_{layer:02d}/op/{name}", tensor, **context)
    attended = model._gqa(
        q,
        k,
        value,
        model.config.full_index(layer),
        start,
    )
    output = linear(
        sigmoid_mul(gate, attended).reshape(-1, model.config.q_size),
        model.weight(attention_weights.output, small_t=small_t),
        small_t=small_t,
    )
    return residual_add(x, output)


def gdn_mixer(model, layer, x, tap, context, *, small_t) -> torch.Tensor:
    _, state = model._ready()
    layer_weights = model.binding.text.layers[layer]
    gdn_weights = layer_weights.gdn
    index = model.config.gdn_index(layer)
    h = rmsnorm(x, model.weight(layer_weights.input_norm), eps=model.config.rms_eps)

    projected = linear(
        h,
        model.block_weight(gdn_weights.query_key_value_z, small_t=small_t),
        small_t=small_t,
    )
    qkv = projected[:, : model.config.conv_dim]
    z = projected[:, model.config.conv_dim :].reshape(
        -1,
        model.config.gdn_v_heads,
        model.config.gdn_v_dim,
    )
    ab = linear(
        h,
        model.block_weight(gdn_weights.a_b_projection, small_t=small_t),
        small_t=small_t,
    )
    a, b = ab.split(model.config.gdn_v_heads, dim=-1)

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
        qkv[:, : model.config.key_dim].reshape(
            -1,
            model.config.gdn_k_heads,
            model.config.gdn_k_dim,
        ),
        eps=model.config.rms_eps,
    )
    k = l2norm(
        qkv[:, model.config.key_dim : 2 * model.config.key_dim].reshape(
            -1,
            model.config.gdn_k_heads,
            model.config.gdn_k_dim,
        ),
        eps=model.config.rms_eps,
    )
    value = qkv[:, 2 * model.config.key_dim :].reshape(
        -1,
        model.config.gdn_v_heads,
        model.config.gdn_v_dim,
    )
    output, state.ssm[index] = gated_delta_net(q, k, value, g, beta, state.ssm[index], scale=model.config.gdn_scale)
    if tap.level == "op":
        for name, tensor in (
            ("conv", qkv),
            ("g", g),
            ("beta", beta),
            ("gdn", output),
        ):
            model._tap(tap, f"layer_{layer:02d}/op/{name}", tensor, **context)
    output = rmsnorm(output, model.weight(gdn_weights.norm), unit_offset=False, z=z, eps=model.config.rms_eps)
    return residual_add(
        x,
        linear(
            output.reshape(-1, model.config.value_dim),
            model.weight(gdn_weights.output, small_t=small_t),
            small_t=small_t,
        ),
    )


def moe_layer(model, layer, x, tap, context, *, small_t) -> torch.Tensor:
    layer_weights = model.binding.text.layers[layer]
    h = rmsnorm(x, model.weight(layer_weights.post_attention_norm), eps=model.config.rms_eps)
    result = sparse_moe(model, layer_weights.moe, h, small_t=small_t)
    if tap.level == "op":
        for name, tensor in (
            ("router_logits", result.router_logits),
            ("route_weights", result.route_weights),
            ("expert_ids", result.expert_ids),
        ):
            model._tap(tap, f"layer_{layer:02d}/op/{name}", tensor, **context)
    return residual_add(x, result.output)


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
    tap = tap or NullTap()
    context = dict(phase=phase, step=step, chunk=chunk, position=start)
    small_t = phase == "decode"
    for layer in range(model.config.layers):
        x = (
            attention_mixer(
                model,
                layer,
                x,
                positions,
                start,
                tap,
                context,
                small_t=small_t,
            )
            if model.config.is_full(layer)
            else gdn_mixer(
                model,
                layer,
                x,
                tap,
                context,
                small_t=small_t,
            )
        )
        model._tap(tap, f"layer_{layer:02d}/mixer", x, **context)
        x = moe_layer(model, layer, x, tap, context, small_t=small_t)
        model._tap(tap, f"layer_{layer:02d}/moe", x, **context)
    return x


__all__ = ["attention_mixer", "gdn_mixer", "moe_layer", "run"]
