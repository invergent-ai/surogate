"""GLM-5.3-Flash (``glm5_next``) mixers and residual plumbing.

Three mechanisms this architecture does not share with anything already in the
DSL, transcribed from ``transformers/models/glm5_next/modular_glm5_next.py``
(the reference implementation) and cross-checked against the released
checkpoint's tensor contract (``study/colibri/c/tools/check_glm53_checkpoint.py``):

``Glm5NextHyperConnection`` / ``Glm5NextHyperConnectionCombine``
    Manifold-constrained hyper-connections (mHC, Xie et al. 2026), the
    DeepSeek-V4 formulation GLM-5.3 inherits. The residual is ``hc_mult``
    parallel streams; one *unweighted* RMSNorm over the flattened ``hc*d_model``
    layout feeds a single projection ``fn`` whose ``(2+H)*H`` outputs split into
    ``pre`` (stream collapse), ``post`` (block-output placement) and ``comb``
    (an ``H x H`` matrix Sinkhorn-projected onto the doubly-stochastic manifold).
    Unlike Qwen3.8-Flash-Next's hyper-connections these do **not** replace the
    layer norms: GLM-5.3 keeps ``input_layernorm`` / ``post_attention_layernorm``
    on the collapsed stream and a real ``model.norm`` after the head.

``Glm5NextKimiDeltaMixer``
    Kimi Delta Attention (KDA): a gated delta rule whose forget gate is
    **per key channel**, not per head — ``g`` has shape ``[B, T, H, D]`` and
    decays the recurrent state along the key axis. That is strictly more
    fine-grained than the Qwen3.5 GDN this DSL already lowers, which is why it
    cannot reuse ``chunk_gated_delta_rule``/``qwen3_5_decay`` (see below).

``Glm5NextLatentAttention``
    DeepSeek-style MLA with ``qk_rope_head_dim == 0`` — NoPE, so there is no
    rotary anywhere in the text stack. Query and key/value both come through a
    low-rank latent with its own RMSNorm; after ``kv_b_proj`` every head is
    materialised, so the dense path is ordinary causal attention.

DEFERRED, DELIBERATELY (no DSL primitive exists and no faithful lowering can be
built out of the ones that do — a wrong lowering is worse than a named hole):

* ``mhc_gates`` — the sigmoid/softmax/Sinkhorn head of the mHC mix. Sinkhorn
  needs 20 alternating row/column *divisions*; the graph builder has no reduce
  and no reciprocal, so this is emitted as one named custom op taking the mix
  logits plus ``base``/``scale`` and returning ``(pre, post, comb)``. The norm
  and the projection that feed it — the expensive parts — are real ops.
  **There is no kernel behind this op name yet.**
* ``kda_decay`` — ``lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))``.
  Structurally the counterpart of the existing fused ``qwen3_5_decay`` op, but a
  different formula on a different (per-channel) layout. **No kernel yet.**
* ``chunk_kimi_delta_rule`` — the KDA core recurrence. Same signature as
  ``chunk_gated_delta_rule`` except ``g`` is ``[B, T, H, D]``. **No kernel yet.**

Everything else here is expressed in existing primitives.
"""

from __future__ import annotations

from typing import Any

from ..dim import B, Dim, T
from ..nn import Module, Proxy, Tracer


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

        # -- params (tiny, never quantised) ---------------------------------
        fn_w = tracer.register_param("fn", (self.MIX, self.SC), quantizable=False)
        base_w = tracer.register_param("base", (self.MIX,), dtype="fp32", quantizable=False)
        scale_w = tracer.register_param("scale", (3,), dtype="fp32", quantizable=False)

        # -- activation slots ------------------------------------------------
        tracer.register_activation("normed_rstd", ("B * T",), dtype="fp32", save=True)
        logits_slot = tracer.register_activation(
            "logits",
            ("B * T", self.MIX),
            save=True,
            share_policy="per_layer",
            description="mHC mix logits (pre | post | comb)",
        )
        pre_slot = tracer.register_activation("pre", ("B * T", self.S), save=True, share_policy="per_layer")
        post_slot = tracer.register_activation("post", ("B * T", self.S), save=True, share_policy="per_layer")
        comb_slot = tracer.register_activation(
            "comb",
            ("B * T", self.S, self.S),
            save=True,
            share_policy="per_layer",
            description="mHC doubly-stochastic stream mixer",
        )
        mixed_slot = tracer.register_activation(
            "mixed",
            ("B", "T", "C"),
            share_policy="per_layer",
            description="mHC collapsed stream (sublayer input)",
        )

        # -- graph -----------------------------------------------------------
        res_flat = g.view(residual.ref, shape=[B * T, self.SC], out_name=tracer.prefixed("res_flat"))

        # Unweighted RMSNorm over the whole hc*d_model layout: rmsnorm with a
        # gamma of ones is exactly DeepseekV4UnweightedRMSNorm.
        ones_ref = g.ones(shape=[self.SC], dtype="bf16")
        normed, _rstd = g.rmsnorm(
            res_flat,
            ones_ref,
            eps=self.eps,
            y_name=tracer.prefixed("normed"),
            rstd_name=tracer.prefixed("normed_rstd"),
        )

        logits = g.matmul(normed, fn_w, transpose="NT", out_name=logits_slot)

        # DEFERRED: bias/scale + sigmoid/softmax + Sinkhorn projection. One
        # named custom op; no kernel implements it yet (see module docstring).
        pre, post, comb = g.custom(
            "mhc_gates",
            logits,
            base_w,
            scale_w,
            num_outputs=3,
            hc_mult=self.S,
            hc_eps=self.hc_eps,
            hc_sinkhorn_iters=self.hc_sinkhorn_iters,
        )
        pre = g.view(pre, shape=[B * T, self.S], out_name=pre_slot)
        post = g.view(post, shape=[B * T, self.S], out_name=post_slot)
        comb = g.view(comb, shape=[B * T, self.S, self.S], out_name=comb_slot)

        # Collapse: sum_s pre[s] * stream[s]. Real ops -- H is 4.
        stream_sizes = [self.d_model] * self.S
        streams = g.split(res_flat, split_size=stream_sizes, dim=1)
        pre_cols = g.split(pre, split_size=[1] * self.S, dim=1)
        acc = g.mul(streams[0], pre_cols[0], out_name=tracer.prefixed("collapse0"))
        for i in range(1, self.S):
            scaled = g.mul(streams[i], pre_cols[i], out_name=tracer.prefixed(f"collapse{i}"))
            acc = g.add(acc, scaled)
        mixed = g.view(acc, shape=[B, T, self.C], out_name=mixed_slot)

        return Proxy(mixed_slot, mixed), Proxy(post_slot, post), Proxy(comb_slot, comb)


class Glm5NextHyperConnectionCombine(Module):
    """mHC combine: place the sublayer output back on the residual streams.

    ``new[s] = post[s] * out + sum_j comb[j, s] * residual[j]`` — the transposed
    ``comb`` matmul of the reference, unrolled over the (four) streams. No
    parameters of its own.

    The unroll is exact but costs ``hc + hc^2`` elementwise ops per site (20 at
    ``hc_mult=4``); a fused ``mhc_combine`` kernel is the obvious replacement
    once one exists. It is written out rather than deferred because, unlike the
    Sinkhorn head, it *can* be said in the primitives the DSL already has.
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

        res_slot = tracer.register_activation(
            "res",
            ("B", "T", self.SC),
            share_policy="per_layer",
            description="Residual streams after the mHC combine",
        )

        res_flat = g.view(residual.ref, shape=[B * T, self.SC], out_name=tracer.prefixed("res_flat"))
        out_flat = g.view(out.ref, shape=[B * T, self.C], out_name=tracer.prefixed("out_flat"))
        comb_flat = g.view(comb.ref, shape=[B * T, self.S * self.S], out_name=tracer.prefixed("comb_flat"))

        stream_sizes = [self.d_model] * self.S
        res_parts = g.split(res_flat, split_size=stream_sizes, dim=1)
        post_cols = g.split(post.ref, split_size=[1] * self.S, dim=1)
        comb_cols = g.split(comb_flat, split_size=[1] * (self.S * self.S), dim=1)

        combined = []
        for s in range(self.S):
            acc = g.mul(out_flat, post_cols[s], out_name=tracer.prefixed(f"place{s}"))
            for j in range(self.S):
                mixed_j = g.mul(res_parts[j], comb_cols[j * self.S + s])
                acc = g.add(acc, mixed_j)
            combined.append(acc)

        cat = g.concat(*combined, dim=1, split_size=stream_sizes)
        res_out = g.view(cat, shape=[B, T, self.SC], out_name=res_slot)
        return Proxy(res_slot, res_out)


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

    ``kda_decay`` and ``chunk_kimi_delta_rule`` are declared-but-unimplemented
    custom ops — see the module docstring.
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
        (x,) = args

        # -- params ----------------------------------------------------------
        qkv_w = tracer.register_param("qkv_weight", (self.ConvDim, "C"))
        tracer.register_param("conv_weight", (self.ConvDim, 1, self.ConvK), quantizable=False)
        f_a_w = tracer.register_param("f_a_weight", (self.Dh, "C"), quantizable=False)
        f_b_w = tracer.register_param("f_b_weight", (self.QKVDim, self.Dh), quantizable=False)
        tracer.register_param("dt_bias", (self.QKVDim,), dtype="fp32", quantizable=False)
        tracer.register_param("A_log", (self.H,), dtype="fp32", quantizable=False)
        b_w = tracer.register_param("b_weight", (self.H, "C"), quantizable=False)
        g_a_w = tracer.register_param("g_a_weight", (self.Dh, "C"), quantizable=False)
        g_b_w = tracer.register_param("g_b_weight", (self.QKVDim, self.Dh), quantizable=False)
        tracer.register_param("o_norm_weight", (self.Dh,), quantizable=False)
        out_w = tracer.register_param("out_weight", ("C", self.QKVDim))

        # -- activation slots ------------------------------------------------
        out_slot = tracer.register_activation(
            "out",
            ("B", "T", "C"),
            share_policy="per_layer",
            description="KDA mixer output",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(x.ref, shape=[B * T, self.C], out_name=tracer.prefixed("x_flat"))

        mixed_qkv_flat = g.matmul(
            x_flat, qkv_w, transpose="NT", out_name=tracer.prefixed("mixed_qkv_flat")
        )
        mixed_qkv = g.view(
            mixed_qkv_flat, shape=[B, T, self.ConvDim], out_name=tracer.prefixed("mixed_qkv")
        )
        mixed_qkv_cf = g.transpose(mixed_qkv, dim0=1, dim1=2)
        conv_w2d = g.view(
            tracer.prefixed("conv_weight"),
            shape=[self.ConvDim, self.ConvK],
            out_name=tracer.prefixed("conv_w2d"),
        )
        conv_out_cf = g.mamba_conv1d(
            mixed_qkv_cf,
            conv_w2d,
            None,
            activation="silu",
            out_name=tracer.prefixed("conv_out_cf"),
        )
        conv_out = g.transpose(conv_out_cf, dim0=1, dim1=2)

        q_flat, k_flat, v_flat = g.split(
            conv_out, split_size=[self.QKVDim, self.QKVDim, self.QKVDim], dim=2
        )
        query = g.view(q_flat, shape=[B, T, self.H, self.Dh], out_name=tracer.prefixed("query"))
        key = g.view(k_flat, shape=[B, T, self.H, self.Dh], out_name=tracer.prefixed("key"))
        value = g.view(v_flat, shape=[B, T, self.H, self.Dh], out_name=tracer.prefixed("value"))

        # Forget gate: low-rank projection, then the fused decay.
        f_a = g.matmul(x_flat, f_a_w, transpose="NT", out_name=tracer.prefixed("f_a"))
        f_b = g.matmul(f_a, f_b_w, transpose="NT", out_name=tracer.prefixed("f_b"))
        f_b = g.view(f_b, shape=[B, T, self.H, self.Dh], out_name=tracer.prefixed("f_gate"))
        # DEFERRED: no kernel implements `kda_decay` yet (module docstring).
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

        # DEFERRED: no kernel implements `chunk_kimi_delta_rule` yet. It differs
        # from `chunk_gated_delta_rule` only in that `decay` is per key channel.
        core_attn_out, _state = g.custom(
            "chunk_kimi_delta_rule",
            query,
            key,
            value,
            decay,
            beta,
            num_outputs=2,
            scale=0.0,
            chunk_size=self.chunk_size,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        # Output gate: low-rank, then the sigmoid-gated per-head RMSNorm.
        g_a = g.matmul(x_flat, g_a_w, transpose="NT", out_name=tracer.prefixed("g_a"))
        g_b = g.matmul(g_a, g_b_w, transpose="NT", out_name=tracer.prefixed("g_b"))

        core_flat = g.view(
            core_attn_out, shape=[B * T * self.H, self.Dh], out_name=tracer.prefixed("core_flat")
        )
        gate_flat = g.view(
            g_b, shape=[B * T * self.H, self.Dh], out_name=tracer.prefixed("gate_flat")
        )
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
        gated = g.view(
            gated_flat, shape=[B * T, self.QKVDim], out_name=tracer.prefixed("gated")
        )
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

    DEFERRED: the DSA indexer (``wq_b``/``wk``/``k_norm``/``weights_proj`` plus
    the k-pool compression tensors) and its top-k masking. Training runs dense
    causal attention, which is the exact semantics whenever the sequence fits
    inside ``index_topk``; the reference model likewise trains dense. The
    indexer tensors are simply not declared here, mirroring how ``qwen4_exp``
    treats its QSA indexer.
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
    ) -> None:
        super().__init__()
        if qk_rope_head_dim != 0:
            raise ValueError(
                "Glm5NextLatentAttention is the NoPE variant: qk_rope_head_dim must be 0, "
                f"got {qk_rope_head_dim}"
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

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (x,) = args

        # -- params ----------------------------------------------------------
        q_a_w = tracer.register_param("q_a_weight", (self.QRank, "C"))
        tracer.register_param("q_a_norm_weight", (self.QRank,), quantizable=False)
        q_b_w = tracer.register_param("q_b_weight", (self.QDim, self.QRank))
        kv_a_w = tracer.register_param("kv_a_weight", (self.KVRank, "C"))
        tracer.register_param("kv_a_norm_weight", (self.KVRank,), quantizable=False)
        kv_b_w = tracer.register_param("kv_b_weight", (self.KVBDim, self.KVRank))
        out_w = tracer.register_param("out_weight", ("C", self.VDim))

        # -- activation slots ------------------------------------------------
        tracer.register_activation("q_a_rstd", ("B * T",), dtype="fp32", save=True)
        tracer.register_activation("kv_a_rstd", ("B * T",), dtype="fp32", save=True)
        q_resid_slot = tracer.register_activation(
            "q_resid",
            ("B * T", self.QRank),
            save=True,
            share_policy="per_layer",
            description="Query latent (also the DSA indexer's input, when it lands)",
        )
        att_slot = tracer.register_activation(
            "att", ("B", "T", self.VDim), save=True, share_policy="always_recompute"
        )
        tracer.register_activation(
            "lse", ("B", self.Hq, "T"), dtype="fp32", save=True, share_policy="always_recompute"
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
        key = g.narrow(kv, dim=3, start=0, length=self.qk_nope_head_dim, out_name=tracer.prefixed("key"))
        key = g.contiguous(key)
        value = g.narrow(
            kv, dim=3, start=self.qk_nope_head_dim, length=self.VHead, out_name=tracer.prefixed("value")
        )
        value = g.contiguous(value)

        attn_out, _lse = g.flash_attention_qkv(
            query,
            key,
            value,
            causal=True,
            softmax_scale=self.softmax_scale,
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
        )

        att_flat = g.view(attn_out, shape=[B * T, self.VDim], out_name=tracer.prefixed("att_flat"))
        out_flat = g.matmul(att_flat, out_w, transpose="NT", out_name=tracer.prefixed("att_out_flat"))
        out = g.view(out_flat, shape=[B, T, self.C], out_name=out_slot)
        return Proxy(out_slot, out)
