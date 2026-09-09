"""Hyper-connections: the residual path of Qwen3.8-Flash-Next (qwen4_exp).

Instead of one residual stream with a norm in front of each sublayer, the model carries
``hc_count`` parallel residual streams (4 for Flash-Next, so the residual is
``hc_count * d_model`` wide) and wraps every sublayer in a mix/combine pair:

    mixed, inject = mix(residual)          # collapse the streams into one sublayer input
    out           = sublayer(mixed)
    residual      = combine(residual, out, inject)   # scatter the output back, per stream

There is no ``input_layernorm``, ``post_attention_layernorm`` or final ``norm`` in the
checkpoint — ``mix`` is where the normalisation lives, and the final ``mix`` before the
LM head (``include_inject=False``) *is* the output norm.

The algebra (transcribed from the serve engine's ``ops::hyper_connection`` and the
llama.cpp ``build_hc_mix``/``build_hc_combine`` reference, cross-checked against
``study/FreeToken/.../qwen4exp/hyper_connections.py``):

    n      = per-stream RMSNorm(residual) * gamma          # gamma spans all hc*D channels
    lo     = silu((down @ n) / hc)                         # the 1/hc precedes the SiLU
    gate   = sigmoid(up @ lo)
    mixed  = mean over streams of (gate * n)               # mean, not sum
    inject = 2 * sigmoid((inject_w @ n) / hc)              # centred on 1: zero inject ==
    combine: residual[s] += inject[s] * out                # an ordinary residual add

The RMS reduction runs over ONE stream (d_model channels) while gamma spans the whole
``hc * d_model`` layout — a grouped norm. The HF checkpoint stores ``hc_norm.weight``
ZERO-CENTRED like the family's other norms (its name ends in ``norm.weight``, so the
GGUF exporter's inherited ``+1`` fold catches it — which is why FreeToken sees the GGUF
gamma "already 1+w" and multiplies plainly). Loading from HF therefore applies
``weight + 1`` here, exactly like :class:`RMSNormPlus1`.
"""

from __future__ import annotations

from typing import Any

from ..dim import B, Dim, T
from ..nn import Module, Proxy, Tracer


class HyperConnection(Module):
    """One mix: collapse the ``hc`` residual streams into a sublayer input.

    Called with the wide residual ``(B, T, hc*d_model)``; returns
    ``(mixed (B,T,d_model), inject (B*T, hc))`` — or just ``mixed`` when built with
    ``include_inject=False`` (the final mix before the LM head has no inject weights).
    """

    _hf_mapping_defaults_ = {
        "norm": "{prefix}.hc_norm",
        "down": "{prefix}.input_mix_weight_down",
        "up": "{prefix}.input_mix_weight_up",
        "inject": "{prefix}.block_inject_weight",
    }

    def __init__(
        self,
        d_model: int,
        hc_count: int = 4,
        hc_lowrank: int = 320,
        eps: float = 1e-6,
        include_inject: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.include_inject = include_inject
        self.C = Dim("C")
        # Concrete ints for shape resolution.
        self.S = hc_count
        self.R = hc_lowrank
        self.SC = hc_count * d_model

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy | tuple[Proxy, ...]:
        g = tracer.graph
        (residual,) = args

        # -- params (all small; kept out of quantized recipes) ---------------
        tracer.register_param("norm", (self.SC,), quantizable=False)
        tracer.register_param("down", (self.R, self.SC), quantizable=False)
        tracer.register_param("up", (self.SC, self.R), quantizable=False)
        if self.include_inject:
            tracer.register_param("inject", (self.S, self.SC), quantizable=False)

        mixed_slot = tracer.register_activation(
            "mixed",
            ("B", "T", "C"),
            share_policy="per_layer",
            description="Hyper-connection mix output (sublayer input)",
        )

        res_flat = g.view(
            residual.ref,
            shape=[B * T, self.SC],
            out_name=tracer.prefixed("res_flat"),
        )

        # Grouped RMSNorm: reduce per stream, gamma over the whole hc*D layout.
        stream_sizes = [self.d_model] * self.S
        streams = g.split(
            res_flat,
            split_size=stream_sizes,
            dim=1,
            out_names=[tracer.prefixed(f"stream{i}") for i in range(self.S)],
        )
        # HF stores the gamma zero-centred; effective gain is 1 + w (see module docstring).
        ones_ref = g.ones(shape=[self.SC], dtype="bf16")
        gamma_eff = g.add(tracer.prefixed("norm"), ones_ref, out_name=tracer.prefixed("norm_eff"))
        gammas = g.split(
            gamma_eff,
            split_size=stream_sizes,
            dim=0,
            out_names=[tracer.prefixed(f"gamma{i}") for i in range(self.S)],
        )
        normed_streams = []
        for i in range(self.S):
            tracer.register_activation(
                f"n{i}_rstd",
                ("B * T",),
                dtype="fp32",
                save=True,
            )
            n_i, _ = g.rmsnorm(
                streams[i],
                gammas[i],
                eps=self.eps,
                y_name=tracer.prefixed(f"n{i}"),
                rstd_name=tracer.prefixed(f"n{i}_rstd"),
            )
            normed_streams.append(n_i)
        n = g.concat(
            *normed_streams,
            dim=1,
            split_size=stream_sizes,
            out_name=tracer.prefixed("normed"),
        )

        # Low-rank gate: silu(down @ n / hc) -> sigmoid(up @ .)
        lo = g.matmul(n, tracer.prefixed("down"), transpose="NT", out_name=tracer.prefixed("lo_proj"))
        lo = g.scale(lo, factor=1.0 / self.S)
        lo = g.silu(lo, out_name=tracer.prefixed("lo_act"))
        gate = g.matmul(lo, tracer.prefixed("up"), transpose="NT", out_name=tracer.prefixed("gate_proj"))
        gate = g.sigmoid(gate, out_name=tracer.prefixed("gate"))
        gated = g.mul(n, gate, out_name=tracer.prefixed("gated_norm"))

        # Mean over streams.
        gated_streams = g.split(gated, split_size=stream_sizes, dim=1)
        acc = gated_streams[0]
        for part in gated_streams[1:]:
            acc = g.add(acc, part)
        acc = g.scale(acc, factor=1.0 / self.S)
        mixed = g.view(acc, shape=[B, T, self.C], out_name=mixed_slot)

        if not self.include_inject:
            return Proxy(mixed_slot, mixed)

        # Per-stream scatter weights: 2*sigmoid(inject @ n / hc), centred on 1.
        inject_slot = tracer.register_activation(
            "inject_gates",
            ("B * T", self.S),
            share_policy="per_layer",
            description="Hyper-connection per-stream combine weights",
        )
        ij = g.matmul(n, tracer.prefixed("inject"), transpose="NT", out_name=tracer.prefixed("inject_proj"))
        ij = g.scale(ij, factor=1.0 / self.S)
        ij = g.sigmoid(ij, out_name=tracer.prefixed("inject_sig"))
        ij = g.scale(ij, factor=2.0)
        ij = g.view(ij, shape=[B * T, self.S], out_name=inject_slot)

        return Proxy(mixed_slot, mixed), Proxy(inject_slot, ij)


class HyperConnectionCombine(Module):
    """One combine: scatter a sublayer's output back across the residual streams.

    ``residual[s] += inject[s] * out`` — no parameters of its own; the ``inject``
    weights come from the paired :class:`HyperConnection` mix.
    """

    def __init__(self, d_model: int, hc_count: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.C = Dim("C")
        self.S = hc_count
        self.SC = hc_count * d_model

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        residual, out, inject = args

        res_slot = tracer.register_activation(
            "res",
            ("B", "T", self.SC),
            share_policy="per_layer",
            description="Residual streams after hyper-connection combine",
        )

        res_flat = g.view(
            residual.ref,
            shape=[B * T, self.SC],
            out_name=tracer.prefixed("res_flat"),
        )
        out_flat = g.view(
            out.ref,
            shape=[B * T, self.C],
            out_name=tracer.prefixed("out_flat"),
        )

        stream_sizes = [self.d_model] * self.S
        res_parts = g.split(res_flat, split_size=stream_sizes, dim=1)
        inj_parts = g.split(inject.ref, split_size=[1] * self.S, dim=1)
        combined = []
        for i in range(self.S):
            scaled = g.mul(out_flat, inj_parts[i], out_name=tracer.prefixed(f"scatter{i}"))
            combined.append(g.add(res_parts[i], scaled))
        cat = g.concat(*combined, dim=1, split_size=stream_sizes)
        res_out = g.view(cat, shape=[B, T, self.SC], out_name=res_slot)
        return Proxy(res_slot, res_out)


class StreamBroadcast(Module):
    """Initialise the wide residual: ``hc`` identical copies of the token embedding."""

    def __init__(self, d_model: int, hc_count: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.C = Dim("C")
        self.S = hc_count
        self.SC = hc_count * d_model

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (x,) = args

        res_slot = tracer.register_activation(
            "res",
            ("B", "T", self.SC),
            share_policy="per_layer",
            description="Initial residual streams (embedding broadcast)",
        )
        # Distinct outputs keep concat backward unambiguous. Scale by one is
        # a materialized identity with a native forward and backward kernel.
        copies = [g.scale(x.ref, factor=1.0) for _ in range(self.S)]
        res = g.concat(*copies, dim=2, split_size=[self.d_model] * self.S, out_name=res_slot)
        return Proxy(res_slot, res)
