"""GLM-5.3-Flash text operators.

Native CUDA implements the manifold hyper-connections, bounded KDA decay and
packed causal convolution. BF16 KDA training uses registered, vendored FLA
Triton kernels; standalone FP32 KDA or disabled document masking uses the CUDA
reference recurrence. MLA uses frozen pooled DSA selection and differentiable
sparse attention. Every SwiGLU applies GLM's asymmetric gate/up clamp. Decode
carries convolution, recurrent, indexer and attention history between calls.
"""

from __future__ import annotations

from typing import Any

from ..dim import B, Dim, T
from ..nn import Module, Proxy, Tracer
from ..specs import LoRATarget
from .moe import LagunaMoEExperts


class Glm5NextMoEExperts(LagunaMoEExperts):
    """Use concrete expert shapes to preserve GLM's dense/sparse width changes."""

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        result = super()._trace(tracer, *args, **kwargs)
        dims = {"E": self.num_experts, "C": self.d_model, "M": self.d_ff, "MUp": 2 * self.d_ff}
        for name in ("router_weight", "e_score_correction_bias", "experts_gate_up", "experts_down"):
            param = tracer.params[tracer.prefixed(name)]
            param.shape = tuple(dims.get(dim, dim) for dim in param.shape)
        tracer.params[tracer.prefixed("router_weight")].lora_targets = []
        tracer.params[tracer.prefixed("e_score_correction_bias")].frozen = True
        # GLM routes in FP32. Rounding sigmoid scores to BF16 before top-k
        # creates ties and can select entirely different experts.
        router_slots = {tracer.prefixed(name) for name in ("router_logits", "router_probs", "routing_weights")}
        for slot in tracer.forward_slots:
            if slot.name in router_slots:
                slot.dtype = "fp32"
        return result


class Glm5NextDenseMLP(Module):
    """Dense SwiGLU with explicit dimensions beside the narrower MoE experts."""

    def __init__(self, d_model: int, d_ff: int, limit: float) -> None:
        super().__init__()
        self.d_model, self.d_ff, self.limit = d_model, d_ff, limit

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (x,) = args
        c, m = self.d_model, self.d_ff
        up = tracer.register_param(
            "up_weight",
            (2 * m, c),
            lora_targets=[
                LoRATarget(name="up", size=m),
                LoRATarget(name="gate", offset=m, size=m),
            ],
        )
        down = tracer.register_param("down_weight", (c, m), lora_targets=[LoRATarget(name="down", size=c)])
        up_slot = tracer.register_activation(
            "up", ("B", "T", 2 * m), aliases=["up_flat"], share_policy="when_recomputed"
        )
        act_slot = tracer.register_activation(
            "act", ("B", "T", m), aliases=["act_flat"], share_policy="when_recomputed"
        )
        out_slot = tracer.register_activation("down", ("B", "T", c), aliases=["down_flat"], share_policy="per_layer")
        xf = g.view(x.ref, shape=[B * T, c], out_name=tracer.prefixed("x_flat"))
        projected = g.matmul(xf, up, transpose="NT", out_name=tracer.prefixed("up_flat"))
        projected = g.view(projected, shape=[B, T, 2 * m], out_name=up_slot)
        act = g.swiglu(projected, limit=self.limit, out_name=act_slot)
        act = g.view(act, shape=[B * T, m], out_name=tracer.prefixed("act_flat"))
        result = g.matmul(act, down, transpose="NT", out_name=tracer.prefixed("down_flat"))
        result = g.view(result, shape=[B, T, c], out_name=out_slot)
        return Proxy(out_slot, result)


class Glm5NextHyperConnection(Module):
    """mHC mix: collapse ``hc_mult`` residual streams into one sublayer input.

    Called with the wide residual ``(B, T, hc*d_model)``. Returns
    ``(collapsed (B,T,C), post (B*T,H), comb (B*T,H,H))`` — ``post`` and ``comb``
    are handed to the paired :class:`Glm5NextHyperConnectionCombine`.

    Reference (``DeepseekV4HyperConnection.forward``)::

        flat        = UnweightedRMSNorm(streams.flatten(2))
        pre,post,cb = split(flat @ fn.T, [H, H, H*H])
        pre         = sigmoid(pre * scale[0] + base[:H]) + eps
        post        = 2 * sigmoid(post * scale[1] + base[H:2H])
        comb        = sinkhorn(softmax(cb * scale[2] + base[2H:]))
        collapsed   = (pre[..., None] * streams).sum(stream_axis)

    The collapse reads the RAW streams, not the normalised ones.
    """

    def __init__(
        self,
        d_model: int,
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        hc_sinkhorn_iters: int = 20,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.C = Dim("C")
        # Concrete ints so the IR resolves without a config round-trip.
        self.S = hc_mult
        self.SC = hc_mult * d_model
        self.MIX = (2 + hc_mult) * hc_mult

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> tuple[Proxy, ...]:
        g = tracer.graph
        (residual,) = args
        fn = tracer.register_param("fn", (self.MIX, self.SC), quantizable=False)
        base = tracer.register_param("base", (self.MIX,), dtype="fp32", quantizable=False)
        scale = tracer.register_param("scale", (3,), dtype="fp32", quantizable=False)
        mixed_slot = tracer.register_activation("mixed", ("B", "T", "C"), save=True, share_policy="per_layer")
        post_slot = tracer.register_activation(
            "post", ("B * T", self.S), dtype="fp32", save=True, share_policy="per_layer"
        )
        comb_slot = tracer.register_activation(
            "comb", ("B * T", self.S, self.S), dtype="fp32", save=True, share_policy="per_layer"
        )
        # Norm, projection and stream collapse accumulate in FP32. Keeping them
        # in one operator avoids rounding the mix logits or the pre gates to BF16.
        mixed, post, comb = g.custom(
            "mhc_mix",
            residual.ref,
            fn,
            base,
            scale,
            num_outputs=3,
            hc_mult=self.S,
            hc_eps=self.hc_eps,
            hc_sinkhorn_iters=self.hc_sinkhorn_iters,
            eps=self.eps,
        )
        mixed = g.view(mixed, shape=[B, T, self.C], out_name=mixed_slot)
        post = g.view(post, shape=[B * T, self.S], out_name=post_slot)
        comb = g.view(comb, shape=[B * T, self.S, self.S], out_name=comb_slot)
        return Proxy(mixed_slot, mixed), Proxy(post_slot, post), Proxy(comb_slot, comb)


class Glm5NextHyperConnectionCombine(Module):
    """mHC combine: place the sublayer output back on the residual streams.

    ``new[s] = post[s] * out + sum_j comb[j, s] * residual[j]`` — the transposed
    ``comb`` matmul of the reference, fused over the (four) streams. No
    parameters of its own.
    """

    def __init__(self, d_model: int, hc_mult: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.C = Dim("C")
        self.S = hc_mult
        self.SC = hc_mult * d_model

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        residual, out, post, comb = args
        slot = tracer.register_activation("res", ("B", "T", self.SC), save=True, share_policy="per_layer")
        # Write directly into the declared slot. A view would alias temporary
        # storage, losing the cross-layer residual needed by backward replay.
        combined = g.custom("mhc_combine", residual.ref, out.ref, post.ref, comb.ref, hc_mult=self.S, out_name=slot)
        return Proxy(slot, combined)


class Glm5NextHyperHead(Module):
    """Collapse the residual streams for the LM head: an unweighted mean.

    ``Glm5NextTextHyperHead`` in the reference — note it is a plain mean, unlike
    DeepSeek-V4's weighted head. The model's ``norm`` is applied after it.
    """

    def __init__(self, d_model: int, hc_mult: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.C = Dim("C")
        self.S = hc_mult
        self.SC = hc_mult * d_model

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (residual,) = args

        out_slot = tracer.register_activation(
            "mean",
            ("B", "T", "C"),
            share_policy="per_layer",
            description="Mean over the mHC residual streams",
        )
        res_flat = g.view(residual.ref, shape=[B * T, self.SC], out_name=tracer.prefixed("res_flat"))
        streams = g.split(res_flat, split_size=[self.d_model] * self.S, dim=1)
        acc = streams[0]
        for part in streams[1:]:
            acc = g.add(acc, part)
        acc = g.scale(acc, factor=1.0 / self.S)
        out = g.view(acc, shape=[B, T, self.C], out_name=out_slot)
        return Proxy(out_slot, out)


class Glm5NextKimiDeltaMixer(Module):
    """Kimi Delta Attention (KDA), GLM-5.3's linear-attention mixer.

    Layout, all heads the same width (``linear_num_heads`` x ``linear_head_dim``)::

        qkv    = [q_proj; k_proj; v_proj] @ x     -> depthwise causal conv(k=4) + SiLU
        g      = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))   [B,T,H,D]
        beta   = sigmoid(b_proj(x))                                            [B,T,H]
        core   = kimi_delta_rule(q, k, v, g, beta)     (l2-normed q/k in-kernel)
        out    = o_proj( o_norm(core) * sigmoid(g_b(g_a(x))) )

    The checkpoint stores ``q_proj``/``k_proj``/``v_proj`` and
    ``q_conv1d``/``k_conv1d``/``v_conv1d`` separately; both triples are fused
    here on the dim-0 axis, which is exactly what the reference does at runtime
    (``torch.cat`` then one grouped conv).

    ``kda_decay`` and ``chunk_kimi_delta_rule`` have native forward/backward
    implementations. The latter uses vendored FLA for BF16 with document masking.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 64,
        head_dim: int = 128,
        conv_kernel: int = 4,
        gate_lower_bound: float = -5.0,
        chunk_size: int = 64,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv_kernel = conv_kernel
        self.gate_lower_bound = gate_lower_bound
        self.chunk_size = chunk_size
        self.eps = eps

        self.C = Dim("C")
        # Concrete ints: KDA is uniform, so key/value/query all share H x D.
        self.H = num_heads
        self.Dh = head_dim
        self.QKVDim = num_heads * head_dim
        self.ConvDim = 3 * self.QKVDim
        self.ConvK = conv_kernel

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        x, position_ids = args

        # -- params ----------------------------------------------------------
        qkv_w = tracer.register_param(
            "qkv_weight",
            (self.ConvDim, "C"),
            lora_targets=[
                LoRATarget(name=name, offset=i * self.QKVDim, size=self.QKVDim)
                for i, name in enumerate(("q", "k", "v"))
            ],
        )
        tracer.register_param("conv_weight", (self.ConvDim, 1, self.ConvK), quantizable=False)
        f_a_w = tracer.register_param("f_a_weight", (self.Dh, "C"), quantizable=False)
        f_b_w = tracer.register_param("f_b_weight", (self.QKVDim, self.Dh), quantizable=False)
        tracer.register_param("dt_bias", (self.QKVDim,), dtype="fp32", quantizable=False)
        tracer.register_param("A_log", (self.H,), dtype="fp32", quantizable=False)
        b_w = tracer.register_param("b_weight", (self.H, "C"), quantizable=False)
        g_a_w = tracer.register_param("g_a_weight", (self.Dh, "C"), quantizable=False)
        g_b_w = tracer.register_param("g_b_weight", (self.QKVDim, self.Dh), quantizable=False)
        tracer.register_param("o_norm_weight", (self.Dh,), quantizable=False)
        out_w = tracer.register_param(
            "out_weight", ("C", self.QKVDim), lora_targets=[LoRATarget(name="o", size=self.d_model)]
        )

        # -- activation slots ------------------------------------------------
        out_slot = tracer.register_activation(
            "out",
            ("B", "T", "C"),
            share_policy="per_layer",
            description="KDA mixer output",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(x.ref, shape=[B * T, self.C], out_name=tracer.prefixed("x_flat"))

        mixed_qkv_flat = g.matmul(x_flat, qkv_w, transpose="NT", out_name=tracer.prefixed("mixed_qkv_flat"))
        mixed_qkv = g.view(mixed_qkv_flat, shape=[B, T, self.ConvDim], out_name=tracer.prefixed("mixed_qkv"))
        conv_out = g.custom(
            "glm_causal_conv1d",
            mixed_qkv,
            tracer.prefixed("conv_weight"),
            position_ids.ref,
        )

        q_flat, k_flat, v_flat = g.split(conv_out, split_size=[self.QKVDim, self.QKVDim, self.QKVDim], dim=2)
        query = g.view(q_flat, shape=[B, T, self.H, self.Dh], out_name=tracer.prefixed("query"))
        key = g.view(k_flat, shape=[B, T, self.H, self.Dh], out_name=tracer.prefixed("key"))
        value = g.view(v_flat, shape=[B, T, self.H, self.Dh], out_name=tracer.prefixed("value"))

        # Forget gate: low-rank projection, then the fused decay.
        f_a = g.matmul(x_flat, f_a_w, transpose="NT", out_name=tracer.prefixed("f_a"))
        f_b = g.matmul(f_a, f_b_w, transpose="NT", out_name=tracer.prefixed("f_b"))
        f_b = g.view(f_b, shape=[B, T, self.H, self.Dh], out_name=tracer.prefixed("f_gate"))
        decay = g.custom(
            "kda_decay",
            f_b,
            tracer.prefixed("A_log"),
            tracer.prefixed("dt_bias"),
            lower_bound=self.gate_lower_bound,
        )

        # Input gate (per head).
        b_flat = g.matmul(x_flat, b_w, transpose="NT", out_name=tracer.prefixed("b_flat"))
        b = g.view(b_flat, shape=[B, T, self.H], out_name=tracer.prefixed("b"))
        beta = g.sigmoid(b, out_name=tracer.prefixed("beta"))

        # KDA decay is per key channel, unlike the scalar-head GDN gate.
        core_attn_out = g.custom(
            "chunk_kimi_delta_rule",
            query,
            key,
            value,
            decay,
            beta,
            position_ids.ref,
        )

        # Output gate: low-rank, then the sigmoid-gated per-head RMSNorm.
        g_a = g.matmul(x_flat, g_a_w, transpose="NT", out_name=tracer.prefixed("g_a"))
        g_b = g.matmul(g_a, g_b_w, transpose="NT", out_name=tracer.prefixed("g_b"))

        core_flat = g.view(core_attn_out, shape=[B * T * self.H, self.Dh], out_name=tracer.prefixed("core_flat"))
        gate_flat = g.view(g_b, shape=[B * T * self.H, self.Dh], out_name=tracer.prefixed("gate_flat"))
        gated_flat = g.mamba_gated_rmsnorm(
            core_flat,
            gate_flat,
            tracer.prefixed("o_norm_weight"),
            eps=self.eps,
            n_groups=1,
            norm_before_gate=True,
            gate_activation="sigmoid",
            out_name=tracer.prefixed("gated_flat"),
        )
        gated = g.view(gated_flat, shape=[B * T, self.QKVDim], out_name=tracer.prefixed("gated"))
        out_flat = g.matmul(gated, out_w, transpose="NT", out_name=tracer.prefixed("out_flat"))
        out = g.view(out_flat, shape=[B, T, self.C], out_name=out_slot)
        return Proxy(out_slot, out)


class Glm5NextLatentAttention(Module):
    """GLM-5.3 MLA: DeepSeek multi-head latent attention with NoPE.

    ``qk_rope_head_dim`` is 0 for every released GLM-5.3 config (the config class
    rejects anything else), so ``kv_a_proj_with_mqa`` emits only the latent and
    there is no rotary embedding at all::

        q = q_b( rmsnorm(q_a(x)) ).view(B,T,H,qk_head_dim)
        k,v = split(kv_b( rmsnorm(kv_a(x)) ).view(B,T,H,-1), [qk_nope, v_head])
        out = o_proj( attention(q, k, v, causal, scale=qk_head_dim**-0.5) )

    The frozen DSA indexer selects complete key pools plus the optional current
    tail. Attention gathers only the selected keys; its Q/K/V backward remains
    differentiable. Packed position IDs reset pooling at document boundaries.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
        qk_rope_head_dim: int = 0,
        eps: float = 1e-5,
        index_n_heads: int = 32,
        index_head_dim: int = 128,
        index_kpool: int = 16,
        index_topk: int = 2048,
        index_kpool_always_select_tail: bool = True,
    ) -> None:
        super().__init__()
        if qk_rope_head_dim != 0:
            raise ValueError(
                f"Glm5NextLatentAttention is the NoPE variant: qk_rope_head_dim must be 0, got {qk_rope_head_dim}"
            )
        if q_lora_rank <= 0:
            raise ValueError("Glm5NextLatentAttention requires a positive q_lora_rank")
        if qk_nope_head_dim != v_head_dim:
            # The DSL's attention op takes one head size for Q/K/V. GLM-5.3 has
            # 256 == 256; a checkpoint that split them would need a different op.
            raise ValueError(
                "Glm5NextLatentAttention requires qk_nope_head_dim == v_head_dim "
                f"(got {qk_nope_head_dim} != {v_head_dim})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.eps = eps

        self.C = Dim("C")
        self.Hq = num_heads
        self.QRank = q_lora_rank
        self.KVRank = kv_lora_rank
        self.QKHead = qk_nope_head_dim + qk_rope_head_dim
        self.VHead = v_head_dim
        self.QDim = num_heads * self.QKHead
        self.KVBDim = num_heads * (qk_nope_head_dim + v_head_dim)
        self.VDim = num_heads * v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.softmax_scale = float(self.QKHead) ** -0.5
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_kpool = index_kpool
        self.index_topk = index_topk
        self.index_tail = index_kpool_always_select_tail

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        x, positions = args

        # -- params ----------------------------------------------------------
        q_a_w = tracer.register_param("q_a_weight", (self.QRank, "C"))
        tracer.register_param("q_a_norm_weight", (self.QRank,), quantizable=False)
        q_b_w = tracer.register_param(
            "q_b_weight", (self.QDim, self.QRank), lora_targets=[LoRATarget(name="q", size=self.QDim)]
        )
        kv_a_w = tracer.register_param("kv_a_weight", (self.KVRank, "C"))
        tracer.register_param("kv_a_norm_weight", (self.KVRank,), quantizable=False)
        kv_b_w = tracer.register_param(
            "kv_b_weight", (self.KVBDim, self.KVRank), lora_targets=[LoRATarget(name="k", size=self.KVBDim)]
        )
        out_w = tracer.register_param(
            "out_weight", ("C", self.VDim), lora_targets=[LoRATarget(name="o", size=self.d_model)]
        )
        ih, idim, pool = self.index_n_heads, self.index_head_dim, self.index_kpool
        index_weights = {
            "index_q_weight": (ih * idim, self.QRank),
            "index_k_weight": (idim, "C"),
            "index_head_weight": (ih, "C"),
            "index_compress_weight": (idim, "C"),
            "index_norm_weight": (idim,),
            "index_norm_bias": (idim,),
            "index_ape": (pool, idim),
        }
        for name, shape in index_weights.items():
            tracer.register_param(name, shape, frozen=True, quantizable=False)
        index_slot = tracer.register_activation(
            "indices",
            ("B", "T", self.index_topk + (pool - 1 if self.index_tail else 0)),
            dtype="int32",
            save=True,
            share_policy="per_layer",
        )

        # -- activation slots ------------------------------------------------
        tracer.register_activation("q_a_rstd", ("B * T",), dtype="fp32", save=True)
        tracer.register_activation("kv_a_rstd", ("B * T",), dtype="fp32", save=True)
        q_resid_slot = tracer.register_activation(
            "q_resid",
            ("B * T", self.QRank),
            save=True,
            share_policy="per_layer",
            description="Normalized query latent, shared with the DSA indexer",
        )
        att_slot = tracer.register_activation("att", ("B", "T", self.VDim), save=True, share_policy="always_recompute")
        qkv_slot = tracer.register_activation(
            "qkv", ("B", "T", 3 * self.Hq, self.QKHead), save=True, share_policy="always_recompute"
        )
        out_slot = tracer.register_activation(
            "att_out", ("B", "T", "C"), share_policy="per_layer", description="MLA output"
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(x.ref, shape=[B * T, self.C], out_name=tracer.prefixed("x_flat"))

        q_a = g.matmul(x_flat, q_a_w, transpose="NT", out_name=tracer.prefixed("q_a"))
        q_resid, _ = g.rmsnorm(
            q_a,
            tracer.prefixed("q_a_norm_weight"),
            eps=self.eps,
            y_name=q_resid_slot,
            rstd_name=tracer.prefixed("q_a_rstd"),
        )
        q_flat = g.matmul(q_resid, q_b_w, transpose="NT", out_name=tracer.prefixed("q_flat"))
        query = g.view(q_flat, shape=[B, T, self.Hq, self.QKHead], out_name=tracer.prefixed("query"))

        kv_c = g.matmul(x_flat, kv_a_w, transpose="NT", out_name=tracer.prefixed("kv_latent"))
        kv_n, _ = g.rmsnorm(
            kv_c,
            tracer.prefixed("kv_a_norm_weight"),
            eps=self.eps,
            y_name=tracer.prefixed("kv_normed"),
            rstd_name=tracer.prefixed("kv_a_rstd"),
        )
        kv_flat = g.matmul(kv_n, kv_b_w, transpose="NT", out_name=tracer.prefixed("kv_flat"))
        kv = g.view(
            kv_flat,
            shape=[B, T, self.Hq, self.qk_nope_head_dim + self.VHead],
            out_name=tracer.prefixed("kv"),
        )
        key, value = g.split(kv, dim=3, split_size=[self.qk_nope_head_dim, self.VHead])

        qkv = g.concat(query, key, value, dim=2, split_size=[self.Hq] * 3, out_name=qkv_slot)
        index_q = g.matmul(q_resid, tracer.prefixed("index_q_weight"), transpose="NT")
        index_q = g.view(index_q, shape=[B, T, ih, idim])
        index_k = g.view(g.matmul(x_flat, tracer.prefixed("index_k_weight"), transpose="NT"), shape=[B, T, idim])
        index_gate = g.view(
            g.matmul(x_flat, tracer.prefixed("index_compress_weight"), transpose="NT"), shape=[B, T, idim]
        )
        index_head = g.view(g.matmul(x_flat, tracer.prefixed("index_head_weight"), transpose="NT"), shape=[B, T, ih])
        indices = g.custom(
            "glm_dsa_indexer",
            index_q,
            index_k,
            index_gate,
            index_head,
            tracer.prefixed("index_norm_weight"),
            tracer.prefixed("index_norm_bias"),
            tracer.prefixed("index_ape"),
            positions.ref,
            out_name=index_slot,
            index_size=self.index_topk + (pool - 1 if self.index_tail else 0),
        )
        # Decode stores the normalized latent and expands only selected keys.
        # Training differentiates through qkv; the extra inputs are cache-only.
        attn_out = g.custom("glm_dsa_attention", qkv, indices, kv_n, kv_b_w, out_name=att_slot)

        att_flat = g.view(attn_out, shape=[B * T, self.VDim], out_name=tracer.prefixed("att_flat"))
        out_flat = g.matmul(att_flat, out_w, transpose="NT", out_name=tracer.prefixed("att_out_flat"))
        out = g.view(out_flat, shape=[B, T, self.C], out_name=out_slot)
        return Proxy(out_slot, out)
