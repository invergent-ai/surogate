"""DeepSeek-V4 specific runtime modules: shared-KV MQA attention and the
sqrt-softplus MoE.

Both are transcribed from ``transformers/models/deepseek_v4/modular_deepseek_v4.py``
(``DeepseekV4Attention`` / ``DeepseekV4TopKRouter`` + ``DeepseekV4Experts``). Where a
mechanism has no DSL primitive the gap is named in a ``DEFERRED`` comment right where it
would have been emitted, rather than approximated behind a plausible-looking op — see the
block/model docstrings for the consolidated list.
"""

from __future__ import annotations

from typing import Any

from ..dim import B, Dim, T
from ..nn import Module, Proxy, Tracer
from ..specs import LoRATarget


class DeepseekV4Attention(Module):
    """Shared-KV MQA with a per-head sink and a grouped low-rank output projection.

    Reference (``DeepseekV4Attention.forward``)::

        q_residual = q_a_norm(q_a_proj(x))                  # [B, T, q_lora_rank]
        q          = q_b_norm(q_b_proj(q_residual))         # unweighted RMSNorm per head
        q          = rope(q)
        kv         = rope(kv_norm(kv_proj(x)))              # ONE head; K and V are the
                                                            # same tensor, so V is roped
        o          = attn(q, kv, kv, sinks=sinks, window=sliding_window)
        o          = rope(o, cos, -sin)                     # conjugate rotation
        o          = o_b_proj(o_a_proj(o))                  # grouped, then mixed

    Expressed faithfully here: every projection and norm (shapes and order), the single
    shared KV head, the per-head sink, the 128-token sliding window, and the block-diagonal
    grouped output projection (a batched GEMM over ``o_groups`` head groups followed by the
    ``o_b_proj`` mix).

    DEFERRED — no DSL primitive, see the block docstring:

    * **RoPE convention.** V4 rotates the *trailing* ``qk_rope_head_dim`` channels of each
      head with *interleaved* pairs ``(2i, 2i+1)``; ``kernels/rope.cu`` rotates the
      *leading* ``rotary_dim`` channels pairing ``(i, i + rotary_dim/2)``. Two independent
      mismatches, and the checkpoint's head layout is ``[nope | rope]``. Emitted anyway so
      the skeleton has a positional signal, but it is NOT V4's rotation.
    * **Inverse RoPE on the attention output.** Because K and V are one tensor, the values
      carry RoPE and V4 undoes it on the output with ``(cos, -sin)`` at the query position.
      ``g.rope`` has no conjugate mode, so this step is simply absent.
    * **Long-range compressor branch.** CSA/HCA layers concatenate compressed KV entries
      (and, for CSA, a Lightning-Indexer top-k block mask) onto the KV axis *inside* the
      attention. Neither the compressor's running-window state nor the indexer's top-k
      gather is expressible, so every layer here runs pure sliding-window attention. The
      compressor/indexer tensors are declared as serve objects on the block schema but are
      not loaded as parameters.
    """

    _hf_mapping_defaults_ = {
        "q_a_proj_weight": "{prefix}.q_a_proj.weight",
        "q_a_norm_weight": "{prefix}.q_a_norm.weight",
        "q_b_proj_weight": "{prefix}.q_b_proj.weight",
        "kv_proj_weight": "{prefix}.kv_proj.weight",
        "kv_norm_weight": "{prefix}.kv_norm.weight",
        "o_a_proj_weight": "{prefix}.o_a_proj.weight",
        "o_b_proj_weight": "{prefix}.o_b_proj.weight",
        "sinks": "{prefix}.sinks",
    }

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        head_size: int,
        q_lora_rank: int,
        o_groups: int,
        o_lora_rank: int,
        max_seq: int,
        sliding_window: int = 128,
        rotary_dim: int = 64,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if (num_query_heads * head_size) % o_groups != 0:
            raise ValueError(
                "DeepseekV4Attention requires num_attention_heads * head_dim to be divisible "
                f"by o_groups (got {num_query_heads} * {head_size} % {o_groups})"
            )
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.head_size = head_size
        self.q_lora_rank = q_lora_rank
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.max_seq = max_seq
        self.sliding_window = sliding_window
        self.rotary_dim = rotary_dim
        self.eps = eps

        self.C = Dim("C")
        # Concrete ints: V4 is a hybrid stack whose layer types all share one geometry,
        # so numeric dims keep the block-local shape env free of collisions.
        self.attn_dim = num_query_heads * head_size
        self.o_a_in = self.attn_dim // o_groups
        self.o_a_out = o_groups * o_lora_rank

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        x, position_ids = args

        _hq = self.num_query_heads
        _d = self.head_size
        _qlora = self.q_lora_rank
        _attn = self.attn_dim

        # -- params ----------------------------------------------------------
        q_a_w = tracer.register_param(
            "q_a_proj_weight", (_qlora, "C"), lora_targets=[LoRATarget(name="q_a", size=_qlora)]
        )
        tracer.register_param("q_a_norm_weight", (_qlora,), quantizable=False)
        q_b_w = tracer.register_param(
            "q_b_proj_weight", (_attn, _qlora), lora_targets=[LoRATarget(name="q_b", size=_attn)]
        )
        kv_w = tracer.register_param("kv_proj_weight", (_d, "C"), lora_targets=[LoRATarget(name="kv", size=_d)])
        tracer.register_param("kv_norm_weight", (_d,), quantizable=False)
        o_a_w = tracer.register_param(
            "o_a_proj_weight",
            (self.o_a_out, self.o_a_in),
            lora_targets=[LoRATarget(name="o_a", size=self.o_a_out)],
        )
        o_b_w = tracer.register_param(
            "o_b_proj_weight", ("C", self.o_a_out), lora_targets=[LoRATarget(name="o_b", size=self.d_model)]
        )
        # Per-head sink logit; HF keeps it in fp32 (`_keep_in_fp32_modules_strict`).
        tracer.register_param("sinks", (_hq,), quantizable=False)
        tracer.register_param(
            "rope_freqs",
            (self.max_seq, self.rotary_dim // 2, 2),
            dtype="fp32",
            frozen=True,
            quantizable=False,
        )

        # -- activation slots ------------------------------------------------
        tracer.register_activation("q_residual", ("B * T", _qlora), save=True, share_policy="when_recomputed")
        qkv_slot = tracer.register_activation(
            "qkv_rope", ("B", "T", "QKV"), save=True, share_policy="always_recompute"
        )
        att_slot = tracer.register_activation(
            "att", ("B", "T", _attn), aliases=["att_flat"], save=True, share_policy="always_recompute"
        )
        tracer.register_activation(
            "lse", ("B", _hq, "T"), dtype="fp32", save=True, share_policy="always_recompute"
        )
        att_out_slot = tracer.register_activation(
            "att_out", ("B", "T", "C"), aliases=["att_out_flat"], share_policy="when_recomputed"
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(x.ref, shape=[B * T, self.C], out_name=tracer.prefixed("x_flat"))

        # Query: low-rank down, weighted RMSNorm, up, then an UNWEIGHTED per-head RMSNorm.
        q_a = g.matmul(x_flat, q_a_w, transpose="NT", out_name=tracer.prefixed("q_a_proj"))
        q_residual, _ = g.rmsnorm(
            q_a,
            tracer.prefixed("q_a_norm_weight"),
            eps=self.eps,
            y_name=tracer.prefixed("q_residual"),
            rstd_name=tracer.prefixed("q_a_rstd"),
        )
        tracer.register_activation("q_a_rstd", ("B * T",), dtype="fp32", save=True)
        q = g.matmul(q_residual, q_b_w, transpose="NT", out_name=tracer.prefixed("q_b_proj"))
        # `q_b_norm` is DeepseekV4UnweightedRMSNorm: RMSNorm with no learned gain. A ones
        # vector is that exactly, not an approximation.
        q_rows = g.view(q, shape=[B * T * _hq, _d], out_name=tracer.prefixed("q_rows"))
        ones_d = g.ones(shape=[_d], dtype="bf16")
        tracer.register_activation("q_rstd", ("B * T * " + str(_hq),), dtype="fp32", save=True)
        q_normed, _ = g.rmsnorm(
            q_rows,
            ones_d,
            eps=self.eps,
            y_name=tracer.prefixed("q_normed"),
            rstd_name=tracer.prefixed("q_rstd"),
        )
        q_4d = g.view(q_normed, shape=[B, T, _hq, _d], out_name=tracer.prefixed("q_4d"))

        # Shared KV: one head, and K and V are literally the same tensor.
        kv = g.matmul(x_flat, kv_w, transpose="NT", out_name=tracer.prefixed("kv_proj"))
        tracer.register_activation("kv_rstd", ("B * T",), dtype="fp32", save=True)
        kv_normed, _ = g.rmsnorm(
            kv,
            tracer.prefixed("kv_norm_weight"),
            eps=self.eps,
            y_name=tracer.prefixed("kv_normed"),
            rstd_name=tracer.prefixed("kv_rstd"),
        )
        kv_4d = g.view(kv_normed, shape=[B, T, 1, _d], out_name=tracer.prefixed("kv_4d"))
        # Distinct copy node so the concat backward stays unambiguous (same reason
        # StreamBroadcast copies per stream).
        v_4d = g.copy(kv_4d)
        qkv = g.concat(q_4d, kv_4d, v_4d, dim=2, split_size=[_hq, 1, 1])

        # DEFERRED: leading-half rotation, not V4's trailing-interleaved rotation.
        qkv_rope = g.rope(
            qkv,
            tracer.prefixed("rope_freqs"),
            position_ids.ref,
            rotary_dim=self.rotary_dim,
            out_name=qkv_slot,
        )

        att, _ = g.flash_attention(
            qkv_rope,
            causal=True,
            softmax_scale=float(_d) ** -0.5,
            window_size=self.sliding_window,
            sinks=tracer.prefixed("sinks"),
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
        )

        # DEFERRED: the conjugate `rope(att, cos, -sin)` that strips RoPE off the values
        # belongs here. `g.rope` has no inverse mode.

        # Grouped output projection: `o_groups` independent blocks, each mapping
        # `num_heads*head_dim/o_groups -> o_lora_rank`, then one mix to hidden_size.
        att_flat = g.view(att, shape=[B * T, self.o_groups, self.o_a_in], out_name=tracer.prefixed("att_grouped"))
        att_g = g.contiguous(g.transpose(att_flat, dim0=0, dim1=1))  # [G, B*T, o_a_in]
        o_a_3d = g.view(o_a_w, shape=[self.o_groups, self.o_lora_rank, self.o_a_in])
        grouped = g.batched_matmul(att_g, o_a_3d, transpose="NT")  # [G, B*T, o_lora_rank]
        grouped = g.contiguous(g.transpose(grouped, dim0=0, dim1=1))  # [B*T, G, o_lora_rank]
        grouped = g.view(grouped, shape=[B * T, self.o_a_out], out_name=tracer.prefixed("o_a_proj"))

        out_flat = g.matmul(grouped, o_b_w, transpose="NT", out_name=tracer.prefixed("att_out_flat"))
        out = g.view(out_flat, shape=[B, T, self.C], out_name=att_out_slot)
        return Proxy(att_out_slot, out)


class DeepseekV4MoEExperts(Module):
    """DeepSeek-V4 routed MoE: score → top-k on (score + correction bias) → SwiGLU experts.

    Reference (``DeepseekV4TopKRouter`` + ``DeepseekV4Experts``)::

        scores  = sqrt(softplus(x @ gate.weight.T))
        indices = topk(scores + e_score_correction_bias, top_k)
        weights = scores.gather(indices); weights /= weights.sum() + 1e-20
        weights *= routed_scaling_factor
        y       = sum_k weights_k * down_k(silu(clamp(gate_k, max=L)) * clamp(up_k, -L, L))

    Expressed faithfully: the batched expert layout ``gate_up_proj [E, 2I, C]`` /
    ``down_proj [E, C, I]``, the aux-loss-free correction bias used for *selection only*,
    the unconditional renormalisation over the winners, and ``routed_scaling_factor``
    folded into the routing weights by ``moe_topk``.

    DEFERRED:

    * **``sqrt`` of the router score.** ``sqrtsoftplus`` is ``softplus(x).sqrt()``; the DSL
      has ``softplus`` (a real registered op with autodiff) but no elementwise ``sqrt``, so
      the emitted score is ``softplus(logits)`` — the exact deviation is the missing square
      root, which changes both the bias-shifted top-k selection and the weight magnitudes.
      The schema still declares ``kind="topk_sqrtsoftplus"``, i.e. what the *model* is.
    * **``swiglu_limit`` clamps.** V4 clamps the gate above and the up both ways at 10.0
      before the SiLU; ``kernels/swiglu.cu`` has no limit. A stability guard rather than a
      shaping non-linearity, but it is a real difference.

    The hash-routed leading layers replace the learned top-k with a frozen
    ``tid2eid[input_ids]`` lookup; see :class:`~surogate.dsl.blocks.deepseek_v4` for why
    that is not expressible and what this module does instead.
    """

    _hf_mapping_defaults_ = {
        "router_weight": "{prefix}.gate.weight",
        "e_score_correction_bias": "{prefix}.gate.e_score_correction_bias",
        # DeepSeek-V4 ships the experts pre-batched, but with the halves in the opposite
        # order to surogate's fused layout (see the block mappings for the transform).
        "experts_gate_up": "{prefix}.experts.gate_up_proj",
        "experts_down": "{prefix}.experts.down_proj",
    }

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        num_experts_per_tok: int = 6,
        routed_scaling_factor: float = 1.0,
        swiglu_limit: float = 10.0,
        hash_routed: bool = False,
        ep_size: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_limit = swiglu_limit
        self.hash_routed = hash_routed
        self.ep_size = ep_size

        self.C = Dim("C")
        self.M = Dim("M")
        self.E = Dim("E")
        self.K = Dim("K")
        self.MUp = 2 * self.M

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (x,) = args  # [B*T, C]

        _ff = self.d_ff
        _hidden = self.d_model
        _n_experts = self.num_experts

        # -- params ----------------------------------------------------------
        tracer.register_param(
            "router_weight",
            ("E", "C"),
            quantizable=False,
            lora_targets=[LoRATarget(name="router", size=_n_experts)],
        )
        if self.hash_routed:
            # Frozen token-id -> expert-id table. Declared so the checkpoint tensor has a
            # home and the layer is identifiable as hash-routed; the graph below cannot
            # consume it (see the class docstring).
            tracer.register_param(
                "tid2eid",
                ("Vocab", "K"),
                dtype="int32",
                frozen=True,
                quantizable=False,
            )
        else:
            tracer.register_param(
                "e_score_correction_bias",
                ("E",),
                dtype="fp32",
                frozen=True,
                quantizable=False,
            )
        tracer.register_param(
            "experts_gate_up",
            ("E", "MUp", "C"),
            offload_group="moe_experts",
            lora_targets=[LoRATarget(name="expert_gate_up", size=2 * _ff, grouped=True)],
        )
        tracer.register_param(
            "experts_down",
            ("E", "C", "M"),
            offload_group="moe_experts",
            lora_targets=[LoRATarget(name="expert_down", size=_hidden, grouped=True)],
        )

        # -- activation slots ------------------------------------------------
        for name, shape, dtype in (
            ("router_logits", ("B * T", "E"), None),
            ("router_probs", ("B * T", "E"), None),
            ("routing_weights", ("B * T", "K"), None),
            ("routing_indices", ("B * T", "K"), "int32"),
            ("permuted_input", ("B * T * K", "C"), None),
            ("scatter_indices", ("B * T * K",), "int32"),
            ("expert_gate_up", ("B * T * K", "MUp"), None),
            ("expert_act", ("B * T * K", "M"), None),
            ("expert_down", ("B * T * K", "C"), None),
        ):
            tracer.register_activation(name, shape, dtype=dtype, save=True, share_policy="fft_share")
        if self.ep_size > 1:
            for name, shape, dtype in (
                ("ep_recv_input", ("B * T * K", "C"), None),
                ("ep_recv_scatter", ("B * T * K",), "int32"),
                ("ep_combined", ("B * T * K", "C"), None),
            ):
                tracer.register_activation(
                    name, shape, dtype=dtype, save=True, share_policy="per_layer", when="ep_size > 1"
                )
        out_slot = tracer.register_activation(
            "out", ("B * T", "C"), aliases=["out_flat"], save=True, share_policy="fft_share"
        )

        # -- graph -----------------------------------------------------------
        router_logits = g.matmul(
            x.ref, tracer.prefixed("router_weight"), transpose="NT", out_name=tracer.prefixed("router_logits")
        )
        # DEFERRED: `sqrt(...)` around this softplus (no elementwise sqrt op).
        router_probs = g.softplus(router_logits, out_name=tracer.prefixed("router_probs"))

        # Hash layers select via `tid2eid[input_ids]`; there is no gather-into-routing
        # primitive, so they fall back to the learned selection below. Non-hash layers
        # select on (score + correction bias) and weight by the unbiased score, which is
        # exactly the reference.
        topk_kwargs: dict[str, Any] = {}
        if not self.hash_routed:
            topk_kwargs["correction_bias"] = tracer.prefixed("e_score_correction_bias")
        routing_weights, routing_indices = g.moe_topk(
            router_probs,
            top_k=self.num_experts_per_tok,
            # HF's DeepseekV4TopKRouter renormalises unconditionally — `norm_topk_prob`
            # is in the config but the router never reads it.
            normalize=True,
            scaling_factor=self.routed_scaling_factor,
            weights_name=tracer.prefixed("routing_weights"),
            indices_name=tracer.prefixed("routing_indices"),
            **topk_kwargs,
        )

        permuted_input, scatter_indices = g.moe_permute(
            x.ref,
            routing_indices,
            top_k=self.num_experts_per_tok,
            out_name=tracer.prefixed("permuted_input"),
            scatter_name=tracer.prefixed("scatter_indices"),
        )

        if self.ep_size > 1:
            gemm_input, gemm_scatter = g.ep_dispatch(
                permuted_input,
                routing_indices,
                scatter_indices,
                num_experts=self.num_experts,
                ep_size=self.ep_size,
                top_k=self.num_experts_per_tok,
                out_name=tracer.prefixed("ep_recv_input"),
                recv_scatter_name=tracer.prefixed("ep_recv_scatter"),
            )
        else:
            gemm_input, gemm_scatter = permuted_input, scatter_indices

        expert_gate_up = g.moe_grouped_gemm_gate_up(
            gemm_input,
            tracer.prefixed("experts_gate_up"),
            gemm_scatter,
            out_name=tracer.prefixed("expert_gate_up"),
        )
        # DEFERRED: `swiglu_limit` clamps on gate/up.
        expert_act = g.swiglu(expert_gate_up, out_name=tracer.prefixed("expert_act"))
        expert_down = g.moe_grouped_gemm_down(
            expert_act,
            tracer.prefixed("experts_down"),
            gemm_scatter,
            out_name=tracer.prefixed("expert_down"),
        )

        if self.ep_size > 1:
            expert_down = g.ep_combine(
                expert_down,
                num_experts=self.num_experts,
                ep_size=self.ep_size,
                top_k=self.num_experts_per_tok,
                out_name=tracer.prefixed("ep_combined"),
            )

        moe_out = g.moe_unpermute(
            expert_down,
            routing_weights,
            scatter_indices,
            top_k=self.num_experts_per_tok,
            out_name=out_slot,
        )
        return Proxy(out_slot, moe_out)
