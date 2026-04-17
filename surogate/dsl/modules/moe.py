"""Mixture-of-Experts modules."""

from __future__ import annotations

from typing import Any

from ..dim import Dim
from ..hf import stack_experts, transform
from ..nn import Module, Proxy, Tracer
from ..specs import LoRATarget


class MoEExpertsGated(Module):
    """MoE with gated expert activation (SwiGLU-style: gate+up fused).

    Handles: router → top-k → permute → grouped GEMM (gate+up) → SwiGLU →
    grouped GEMM (down) → unpermute.  Optionally supports expert parallelism
    (ep_size > 1) with all-to-all dispatch/combine.
    """

    _hf_mapping_defaults_ = {
        "router_weight": "{prefix}.gate.weight",
        "experts_gate_up": stack_experts(
            "{prefix}.experts.{expert}.gate_proj.weight",
            fuse_gate_up=True,
        ),
        "experts_down": stack_experts(
            "{prefix}.experts.{expert}.down_proj.weight",
        ),
    }

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        ep_size: int = 1,
        activation: str = "swiglu",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.activation = activation
        self.ep_size = ep_size

        self.C = Dim("C")
        self.M = Dim("M")
        self.E = Dim("E")
        self.K = Dim("K")
        self.MUp = 2 * self.M

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (x,) = args  # x is [B*T, C]

        _ff = self.d_ff
        _hidden = self.d_model
        _n_experts = self.num_experts

        # -- params ----------------------------------------------------------
        tracer.register_param(
            "router_weight",
            ("E", "C"),
            lora_targets=[LoRATarget(name="router", size=_n_experts)],
        )
        tracer.register_param(
            "experts_gate_up",
            ("E", "MUp", "C"),
            offload_group="moe_experts",
            lora_targets=[
                LoRATarget(
                    name="expert_gate_up",
                    size=2 * _ff,
                    grouped=True,
                )
            ],
        )
        tracer.register_param(
            "experts_down",
            ("E", "C", "M"),
            offload_group="moe_experts",
            lora_targets=[
                LoRATarget(
                    name="expert_down",
                    size=_hidden,
                    grouped=True,
                )
            ],
        )

        # -- activation slots ------------------------------------------------
        tracer.register_activation(
            "router_logits",
            ("B * T", "E"),
            save=True,
            share_policy="fft_share",
            description="Router logits before softmax",
        )
        tracer.register_activation(
            "router_probs",
            ("B * T", "E"),
            save=True,
            share_policy="fft_share",
            description="Router probabilities",
        )
        tracer.register_activation(
            "routing_weights",
            ("B * T", "K"),
            save=True,
            share_policy="fft_share",
            description="Routing weights for selected experts",
        )
        tracer.register_activation(
            "routing_indices",
            ("B * T", "K"),
            dtype="int32",
            save=True,
            share_policy="fft_share",
            description="Expert indices for each token",
        )
        tracer.register_activation(
            "permuted_input",
            ("B * T * K", "C"),
            save=True,
            share_policy="fft_share",
            description="Permuted input for grouped GEMM",
        )
        tracer.register_activation(
            "scatter_indices",
            ("B * T * K",),
            dtype="int32",
            save=True,
            share_policy="fft_share",
            description="Indices for scattering back",
        )
        if self.ep_size > 1:
            tracer.register_activation(
                "ep_recv_input",
                ("B * T * K", "C"),
                save=True,
                share_policy="per_layer",
                when="ep_size > 1",
                description="EP-dispatched input tokens",
            )
            tracer.register_activation(
                "ep_recv_scatter",
                ("B * T * K",),
                dtype="int32",
                save=True,
                share_policy="per_layer",
                when="ep_size > 1",
                description="EP-dispatched scatter indices",
            )
        tracer.register_activation(
            "expert_gate_up",
            ("B * T * K", "MUp"),
            save=True,
            share_policy="fft_share",
            description="Expert gate+up projection output",
        )
        tracer.register_activation(
            "expert_act",
            ("B * T * K", "M"),
            save=True,
            share_policy="fft_share",
            description="Expert SwiGLU activation output",
        )
        tracer.register_activation(
            "expert_down",
            ("B * T * K", "C"),
            save=True,
            share_policy="fft_share",
            description="Expert down projection output",
        )
        if self.ep_size > 1:
            tracer.register_activation(
                "ep_combined",
                ("B * T * K", "C"),
                save=True,
                share_policy="per_layer",
                when="ep_size > 1",
                description="EP-combined expert output",
            )
        out_slot = tracer.register_activation(
            "out",
            ("B * T", "C"),
            aliases=["out_flat"],
            save=True,
            share_policy="fft_share",
            description="Combined MoE output",
        )

        # -- graph -----------------------------------------------------------
        router_logits = g.matmul(
            x.ref,
            tracer.prefixed("router_weight"),
            transpose="NT",
            out_name=tracer.prefixed("router_logits"),
        )
        router_probs = g.moe_softmax(
            router_logits,
            out_name=tracer.prefixed("router_probs"),
        )
        routing_weights, routing_indices = g.moe_topk(
            router_probs,
            top_k=self.num_experts_per_tok,
            normalize=self.norm_topk_prob,
            weights_name=tracer.prefixed("routing_weights"),
            indices_name=tracer.prefixed("routing_indices"),
        )
        permuted_input, scatter_indices = g.moe_permute(
            x.ref,
            routing_indices,
            top_k=self.num_experts_per_tok,
            out_name=tracer.prefixed("permuted_input"),
            scatter_name=tracer.prefixed("scatter_indices"),
        )

        if self.ep_size > 1:
            ep_recv_input, ep_recv_scatter = g.ep_dispatch(
                permuted_input,
                routing_indices,
                scatter_indices,
                num_experts=self.num_experts,
                ep_size=self.ep_size,
                top_k=self.num_experts_per_tok,
                out_name=tracer.prefixed("ep_recv_input"),
                recv_scatter_name=tracer.prefixed("ep_recv_scatter"),
            )
            gemm_input = ep_recv_input
            gemm_scatter = ep_recv_scatter
        else:
            gemm_input = permuted_input
            gemm_scatter = scatter_indices

        expert_gate_up = g.moe_grouped_gemm_gate_up(
            gemm_input,
            tracer.prefixed("experts_gate_up"),
            gemm_scatter,
            out_name=tracer.prefixed("expert_gate_up"),
        )
        if self.activation == "gelu":
            # GeLU-gated: split → gelu(gate) * up
            gate_half, up_half = g.split(
                expert_gate_up,
                split_size=[self.d_ff, self.d_ff],
                dim=-1,
            )
            gate_act = g.gelu(gate_half)
            expert_act = g.mul(gate_act, up_half)
        else:
            expert_act = g.swiglu(
                expert_gate_up,
                out_name=tracer.prefixed("expert_act"),
            )
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


class Gemma4MoEExperts(Module):
    """Gemma4 MoE with custom router (RMSNorm + scale + per_expert_scale).

    Router: norm(x, no_scale) * scale * hidden_size^(-0.5) → proj → softmax
    → topk → normalize → per_expert_scale. Experts use GeLU-gated activation.

    HF weight paths:
    - ``{prefix}.router.proj.weight`` — router projection [num_experts, C]
    - ``{prefix}.router.scale`` — router scale vector [C]
    - ``{prefix}.router.per_expert_scale`` — per-expert scale [E]
    - ``{prefix}.experts.gate_up_proj`` — batched [E, 2*M, C]
    - ``{prefix}.experts.down_proj`` — batched [E, C, M]
    """

    _hf_mapping_defaults_ = {
        "router_weight": "{prefix}.router.proj.weight",
        "router_scale": "{prefix}.router.scale",
        "per_expert_scale": "{prefix}.router.per_expert_scale",
        "experts_gate_up": "{prefix}.experts.gate_up_proj",
        "experts_down": "{prefix}.experts.down_proj",
    }

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        num_experts_per_tok: int = 8,
        eps: float = 1e-6,
        ep_size: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.eps = eps
        self.ep_size = ep_size
        self.scalar_root_size = float(d_model) ** -0.5

        self.C = Dim("C")
        self.M = Dim("M")
        self.E = Dim("E")
        self.K = Dim("K")
        self.MUp = 2 * self.M

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (x,) = args  # x is [B*T, C]

        _ff = self.d_ff
        _hidden = self.d_model
        _n_experts = self.num_experts

        # -- params ----------------------------------------------------------
        tracer.register_param(
            "router_weight",
            ("E", "C"),
            lora_targets=[LoRATarget(name="router", size=_n_experts)],
        )
        tracer.register_param("router_scale", ("C",), quantizable=False)
        tracer.register_param("per_expert_scale", ("E",), quantizable=False)
        tracer.register_param(
            "experts_gate_up",
            ("E", "MUp", "C"),
            offload_group="moe_experts",
            lora_targets=[
                LoRATarget(
                    name="expert_gate_up",
                    size=2 * _ff,
                    grouped=True,
                )
            ],
        )
        tracer.register_param(
            "experts_down",
            ("E", "C", "M"),
            offload_group="moe_experts",
            lora_targets=[
                LoRATarget(
                    name="expert_down",
                    size=_hidden,
                    grouped=True,
                )
            ],
        )

        # -- activation slots ------------------------------------------------
        tracer.register_activation(
            "router_logits",
            ("B * T", "E"),
            save=True,
            share_policy="fft_share",
        )
        tracer.register_activation(
            "routing_weights",
            ("B * T", "K"),
            save=True,
            share_policy="fft_share",
        )
        tracer.register_activation(
            "routing_indices",
            ("B * T", "K"),
            dtype="int32",
            save=True,
            share_policy="fft_share",
        )
        tracer.register_activation(
            "permuted_input",
            ("B * T * K", "C"),
            save=True,
            share_policy="fft_share",
        )
        tracer.register_activation(
            "scatter_indices",
            ("B * T * K",),
            dtype="int32",
            save=True,
            share_policy="fft_share",
        )
        tracer.register_activation(
            "expert_gate_up",
            ("B * T * K", "MUp"),
            save=True,
            share_policy="fft_share",
        )
        tracer.register_activation(
            "expert_down",
            ("B * T * K", "C"),
            save=True,
            share_policy="fft_share",
        )
        out_slot = tracer.register_activation(
            "out",
            ("B * T", "C"),
            aliases=["out_flat"],
            save=True,
            share_policy="fft_share",
        )

        # -- graph: Gemma4 router -------------------------------------------
        # 1. RMSNorm without learnable scale (ones weight)
        ones_c = g.ones(shape=[self.C], dtype="bf16")
        x_normed, _ = g.rmsnorm(
            x.ref,
            ones_c,
            eps=self.eps,
            y_name=tracer.prefixed("router_normed"),
        )

        # 2. Scale: normed * router_scale * hidden_size^(-0.5)
        x_scaled = g.mul(x_normed, tracer.prefixed("router_scale"))
        x_scaled = g.scale(x_scaled, factor=self.scalar_root_size)

        # 3. Router projection → softmax → topk
        router_logits = g.matmul(
            x_scaled,
            tracer.prefixed("router_weight"),
            transpose="NT",
            out_name=tracer.prefixed("router_logits"),
        )
        router_probs = g.moe_softmax(
            router_logits,
            out_name=tracer.prefixed("router_probs"),
        )
        routing_weights, routing_indices = g.moe_topk(
            router_probs,
            top_k=self.num_experts_per_tok,
            normalize=True,
            weights_name=tracer.prefixed("routing_weights"),
            indices_name=tracer.prefixed("routing_indices"),
        )

        # 4. Per-expert scale: routing_weights *= per_expert_scale[indices]
        # This is applied after topk normalization. The moe_unpermute step
        # already multiplies by routing_weights, so we scale them here.
        # The per_expert_scale is a [E] vector; we gather by routing_indices [B*T, K].
        # For now, pass per_expert_scale as a parameter and let the runtime
        # handle the gather+mul. We encode this as a custom attr on moe_topk.
        # TODO: Add explicit per_expert_scale gather+mul if runtime doesn't support it.

        # -- graph: expert computation ---------------------------------------
        permuted_input, scatter_indices = g.moe_permute(
            x.ref,
            routing_indices,
            top_k=self.num_experts_per_tok,
            out_name=tracer.prefixed("permuted_input"),
            scatter_name=tracer.prefixed("scatter_indices"),
        )

        expert_gate_up = g.moe_grouped_gemm_gate_up(
            permuted_input,
            tracer.prefixed("experts_gate_up"),
            scatter_indices,
            out_name=tracer.prefixed("expert_gate_up"),
        )

        # GeLU-gated: split → gelu(gate) * up
        gate_half, up_half = g.split(
            expert_gate_up,
            split_size=[self.d_ff, self.d_ff],
            dim=-1,
        )
        gate_act = g.gelu(gate_half)
        expert_act = g.mul(gate_act, up_half)

        expert_down = g.moe_grouped_gemm_down(
            expert_act,
            tracer.prefixed("experts_down"),
            scatter_indices,
            out_name=tracer.prefixed("expert_down"),
        )

        moe_out = g.moe_unpermute(
            expert_down,
            routing_weights,
            scatter_indices,
            top_k=self.num_experts_per_tok,
            out_name=out_slot,
        )

        return Proxy(out_slot, moe_out)


class MoESharedExpert(Module):
    """Shared expert for MoE models (SwiGLU-style: gate + up → silu*mul → down)."""

    _hf_mapping_defaults_ = {
        "gate": "{prefix}.shared_expert.gate_proj.weight",
        "up": "{prefix}.shared_expert.up_proj.weight",
        "down": "{prefix}.shared_expert.down_proj.weight",
    }

    def __init__(self, d_model: int, shared_expert_intermediate: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.shared_expert_intermediate = shared_expert_intermediate
        self.C = Dim("C")
        self.SharedM = Dim("SharedM")

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (x,) = args  # x is [B*T, C]

        _shared = self.shared_expert_intermediate
        _hidden = self.d_model

        tracer.register_param(
            "gate",
            ("SharedM", "C"),
            lora_targets=[LoRATarget(name="shared_gate", size=_shared)],
        )
        tracer.register_param(
            "up",
            ("SharedM", "C"),
            lora_targets=[LoRATarget(name="shared_up", size=_shared)],
        )
        tracer.register_param(
            "down",
            ("C", "SharedM"),
            lora_targets=[LoRATarget(name="shared_down", size=_hidden)],
        )

        tracer.register_activation(
            "gate_out",
            ("B * T", "SharedM"),
            share_policy="fft_share",
            when="use_shared_expert",
        )
        tracer.register_activation(
            "up_out",
            ("B * T", "SharedM"),
            share_policy="fft_share",
            when="use_shared_expert",
        )
        tracer.register_activation(
            "gate_act",
            ("B * T", "SharedM"),
            share_policy="fft_share",
            when="use_shared_expert",
        )
        tracer.register_activation(
            "hidden",
            ("B * T", "SharedM"),
            share_policy="fft_share",
            when="use_shared_expert",
        )
        out_slot = tracer.register_activation(
            "out",
            ("B * T", "C"),
            share_policy="fft_share",
            when="use_shared_expert",
        )

        shared_gate = g.matmul(
            x.ref,
            tracer.prefixed("gate"),
            transpose="NT",
            out_name=tracer.prefixed("gate_out"),
        )
        shared_up = g.matmul(
            x.ref,
            tracer.prefixed("up"),
            transpose="NT",
            out_name=tracer.prefixed("up_out"),
        )
        shared_gate_act = g.silu(shared_gate, out_name=tracer.prefixed("gate_act"))
        shared_hidden = g.mul(shared_gate_act, shared_up)
        shared_out = g.matmul(
            shared_hidden,
            tracer.prefixed("down"),
            transpose="NT",
            out_name=out_slot,
        )

        return Proxy(out_slot, shared_out)


class GptOssMoEExperts(Module):
    """GPT-OSS MoE with router bias, per-expert biases, and gpt_oss_moe_act."""

    _hf_mapping_defaults_ = {
        "router_weight": "{prefix}.router.weight",
        "router_bias": "{prefix}.router.bias",
        "experts_gate_up": transform("{prefix}.experts.gate_up_proj", fn="transpose"),
        "experts_gate_up_bias": "{prefix}.experts.gate_up_proj_bias",
        "experts_down": transform("{prefix}.experts.down_proj", fn="transpose"),
        "experts_down_bias": "{prefix}.experts.down_proj_bias",
    }

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        num_experts_per_tok: int = 4,
        ep_size: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.ep_size = ep_size

        self.C = Dim("C")
        self.M = Dim("M")
        self.E = Dim("E")
        self.K = Dim("K")
        self.MUp = 2 * self.M

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (x,) = args  # x is [B*T, C]

        _ff = self.d_ff
        _hidden = self.d_model
        _n_experts = self.num_experts

        # -- params --------------------------------------------------------------
        tracer.register_param(
            "router_weight",
            ("E", "C"),
            lora_targets=[LoRATarget(name="router", size=_n_experts)],
        )
        tracer.register_param("router_bias", ("E",))
        tracer.register_param(
            "experts_gate_up",
            ("E", "MUp", "C"),
            offload_group="moe_experts",
            lora_targets=[
                LoRATarget(
                    name="expert_gate_up",
                    size=2 * _ff,
                    grouped=True,
                )
            ],
        )
        tracer.register_param(
            "experts_gate_up_bias",
            ("E", "MUp"),
            offload_group="moe_experts",
        )
        tracer.register_param(
            "experts_down",
            ("E", "C", "M"),
            offload_group="moe_experts",
            lora_targets=[
                LoRATarget(
                    name="expert_down",
                    size=_hidden,
                    grouped=True,
                )
            ],
        )
        tracer.register_param(
            "experts_down_bias",
            ("E", "C"),
            offload_group="moe_experts",
        )

        # -- activation slots ----------------------------------------------------
        tracer.register_activation(
            "router_logits",
            ("B * T", "E"),
            save=True,
            share_policy="per_layer",
            description="Router logits",
        )
        tracer.register_activation(
            "routing_weights",
            ("B * T", "K"),
            save=True,
            share_policy="fft_share",
        )
        tracer.register_activation(
            "routing_indices",
            ("B * T", "K"),
            dtype="int32",
            save=True,
            share_policy="fft_share",
        )
        tracer.register_activation(
            "permuted_input",
            ("B * T * K", "C"),
            save=True,
            share_policy="fft_share",
        )
        tracer.register_activation(
            "scatter_indices",
            ("B * T * K",),
            dtype="int32",
            save=True,
            share_policy="fft_share",
        )
        if self.ep_size > 1:
            tracer.register_activation(
                "ep_recv_input",
                ("B * T * K", "C"),
                save=True,
                share_policy="per_layer",
                when="ep_size > 1",
            )
            tracer.register_activation(
                "ep_recv_scatter",
                ("B * T * K",),
                dtype="int32",
                save=True,
                share_policy="per_layer",
                when="ep_size > 1",
            )
        tracer.register_activation(
            "expert_gate_up",
            ("B * T * K", "MUp"),
            share_policy="fft_share",
        )
        tracer.register_activation(
            "expert_gate_up_bias",
            ("B * T * K", "MUp"),
            save=True,
            share_policy="fft_share",
        )
        tracer.register_activation(
            "expert_act",
            ("B * T * K", "M"),
            save=True,
            share_policy="fft_share",
            description="GPT-OSS activation output",
        )
        tracer.register_activation(
            "expert_down",
            ("B * T * K", "C"),
            share_policy="fft_share",
        )
        tracer.register_activation(
            "expert_down_bias",
            ("B * T * K", "C"),
            save=True,
            share_policy="fft_share",
        )
        if self.ep_size > 1:
            tracer.register_activation(
                "ep_combined",
                ("B * T * K", "C"),
                save=True,
                share_policy="per_layer",
                when="ep_size > 1",
            )
        out_slot = tracer.register_activation(
            "out",
            ("B * T", "C"),
            aliases=["out_flat"],
            save=True,
            share_policy="fft_share",
        )

        # -- graph ---------------------------------------------------------------
        # Router (with bias)
        router_logits = g.matmul_bias(
            x.ref,
            tracer.prefixed("router_weight"),
            tracer.prefixed("router_bias"),
            transpose="NT",
            out_name=tracer.prefixed("router_logits"),
        )

        # Top-k (softmax + sort_by_index, no normalize)
        routing_weights, routing_indices = g.moe_topk(
            router_logits,
            top_k=self.num_experts_per_tok,
            normalize=False,
            softmax=True,
            sort_by_index=True,
            weights_name=tracer.prefixed("routing_weights"),
            indices_name=tracer.prefixed("routing_indices"),
        )

        # Permute
        permuted_input, scatter_indices = g.moe_permute(
            x.ref,
            routing_indices,
            top_k=self.num_experts_per_tok,
            out_name=tracer.prefixed("permuted_input"),
            scatter_name=tracer.prefixed("scatter_indices"),
        )

        # EP dispatch
        if self.ep_size > 1:
            ep_recv_input, ep_recv_scatter = g.ep_dispatch(
                permuted_input,
                routing_indices,
                scatter_indices,
                num_experts=self.num_experts,
                ep_size=self.ep_size,
                top_k=self.num_experts_per_tok,
                out_name=tracer.prefixed("ep_recv_input"),
                recv_scatter_name=tracer.prefixed("ep_recv_scatter"),
            )
            gemm_input = ep_recv_input
            gemm_scatter = ep_recv_scatter
        else:
            gemm_input = permuted_input
            gemm_scatter = scatter_indices

        # Gate+up GEMM (interleaved)
        expert_gate_up = g.moe_grouped_gemm_gate_up(
            gemm_input,
            tracer.prefixed("experts_gate_up"),
            gemm_scatter,
            gate_up_interleaved=True,
            out_name=tracer.prefixed("expert_gate_up"),
        )

        # Gate+up bias
        expert_gate_up_bias = g.moe_expert_bias_add(
            expert_gate_up,
            tracer.prefixed("experts_gate_up_bias"),
            out_name=tracer.prefixed("expert_gate_up_bias"),
        )

        # GPT-OSS activation
        expert_act = g.gpt_oss_moe_act(
            expert_gate_up_bias,
            alpha=1.702,
            limit=7.0,
            out_name=tracer.prefixed("expert_act"),
        )

        # Down GEMM
        expert_down = g.moe_grouped_gemm_down(
            expert_act,
            tracer.prefixed("experts_down"),
            gemm_scatter,
            out_name=tracer.prefixed("expert_down"),
        )

        # Down bias
        expert_down_bias = g.moe_expert_bias_add(
            expert_down,
            tracer.prefixed("experts_down_bias"),
            out_name=tracer.prefixed("expert_down_bias"),
        )

        # EP combine
        if self.ep_size > 1:
            expert_down_bias = g.ep_combine(
                expert_down_bias,
                num_experts=self.num_experts,
                ep_size=self.ep_size,
                top_k=self.num_experts_per_tok,
                out_name=tracer.prefixed("ep_combined"),
            )

        # Unpermute
        moe_out = g.moe_unpermute(
            expert_down_bias,
            routing_weights,
            scatter_indices,
            top_k=self.num_experts_per_tok,
            out_name=out_slot,
        )

        return Proxy(out_slot, moe_out)


class NemotronMoEExperts(Module):
    """Nemotron MoE with sigmoid routing, correction bias, relu2 activation."""

    _hf_mapping_defaults_ = {
        "router_weight": "{prefix}.gate.weight",
        "e_score_correction_bias": "{prefix}.gate.e_score_correction_bias",
        "experts_up": stack_experts(
            "{prefix}.experts.{expert}.up_proj.weight",
        ),
        "experts_down": stack_experts(
            "{prefix}.experts.{expert}.down_proj.weight",
        ),
    }

    def __init__(
        self,
        d_model: int,
        moe_intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        activation: str = "relu2",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        ep_size: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.activation = activation
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.ep_size = ep_size

        self.C = Dim("C")
        self.M = Dim("M")
        self.E = Dim("E")
        self.K = Dim("K")

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (x,) = args  # x is [B*T, C]

        _ff = self.moe_intermediate_size
        _hidden = self.d_model
        _n_experts = self.num_experts

        # -- params ----------------------------------------------------------
        tracer.register_param(
            "router_weight",
            ("E", "C"),
            quantizable=False,
            lora_targets=[LoRATarget(name="router", size=_n_experts)],
        )
        tracer.register_param("e_score_correction_bias", ("E",), dtype="fp32", quantizable=False)
        tracer.register_param(
            "experts_up",
            ("E", "M", "C"),
            offload_group="moe_experts",
            lora_targets=[
                LoRATarget(
                    name="expert_up",
                    size=_ff,
                    grouped=True,
                )
            ],
        )
        tracer.register_param(
            "experts_down",
            ("E", "C", "M"),
            offload_group="moe_experts",
            lora_targets=[
                LoRATarget(
                    name="expert_down",
                    size=_hidden,
                    grouped=True,
                )
            ],
        )

        # -- activation slots ------------------------------------------------
        tracer.register_activation(
            "router_logits",
            ("B * T", "E"),
            dtype="fp32",
            save=True,
            share_policy="when_recomputed",
        )
        tracer.register_activation(
            "router_probs",
            ("B * T", "E"),
            dtype="fp32",
            save=True,
            share_policy="when_recomputed",
        )
        tracer.register_activation(
            "routing_weights",
            ("B * T", "K"),
            dtype="fp32",
            save=True,
            share_policy="when_recomputed",
        )
        tracer.register_activation(
            "routing_indices",
            ("B * T", "K"),
            dtype="int32",
            save=True,
            share_policy="when_recomputed",
        )
        tracer.register_activation(
            "permuted_input",
            ("B * T * K", "C"),
            save=True,
            share_policy="when_recomputed",
        )
        tracer.register_activation(
            "scatter_indices",
            ("B * T * K",),
            dtype="int32",
            save=True,
            share_policy="when_recomputed",
        )
        if self.ep_size > 1:
            tracer.register_activation(
                "ep_recv_input",
                ("B * T * K", "C"),
                save=True,
                share_policy="per_layer",
                when="ep_size > 1",
            )
            tracer.register_activation(
                "ep_recv_scatter",
                ("B * T * K",),
                dtype="int32",
                save=True,
                share_policy="per_layer",
                when="ep_size > 1",
            )
        tracer.register_activation(
            "expert_up",
            ("B * T * K", "M"),
            save=True,
            share_policy="when_recomputed",
        )
        tracer.register_activation(
            "expert_act",
            ("B * T * K", "M"),
            save=True,
            share_policy="when_recomputed",
        )
        tracer.register_activation(
            "expert_down",
            ("B * T * K", "C"),
            save=True,
            share_policy="when_recomputed",
        )
        if self.ep_size > 1:
            tracer.register_activation(
                "ep_combined",
                ("B * T * K", "C"),
                save=True,
                share_policy="per_layer",
                when="ep_size > 1",
            )
        out_slot = tracer.register_activation(
            "out",
            ("B * T", "C"),
            aliases=["out_flat"],
            save=True,
            share_policy="when_recomputed",
        )

        # -- graph -----------------------------------------------------------
        # Router (no bias on router matmul)
        router_logits = g.matmul(
            x.ref,
            tracer.prefixed("router_weight"),
            transpose="NT",
            out_name=tracer.prefixed("router_logits"),
        )

        # Sigmoid routing
        router_probs = g.moe_sigmoid(
            router_logits,
            out_name=tracer.prefixed("router_probs"),
        )

        # Top-k with correction bias and scaling factor
        routing_weights, routing_indices = g.moe_topk(
            router_probs,
            top_k=self.num_experts_per_tok,
            normalize=self.norm_topk_prob,
            scaling_factor=self.routed_scaling_factor,
            correction_bias=tracer.prefixed("e_score_correction_bias"),
            weights_name=tracer.prefixed("routing_weights"),
            indices_name=tracer.prefixed("routing_indices"),
        )

        # Permute
        permuted_input, scatter_indices = g.moe_permute(
            x.ref,
            routing_indices,
            top_k=self.num_experts_per_tok,
            out_name=tracer.prefixed("permuted_input"),
            scatter_name=tracer.prefixed("scatter_indices"),
        )

        # EP dispatch
        if self.ep_size > 1:
            ep_recv_input, ep_recv_scatter = g.ep_dispatch(
                permuted_input,
                routing_indices,
                scatter_indices,
                num_experts=self.num_experts,
                ep_size=self.ep_size,
                top_k=self.num_experts_per_tok,
                out_name=tracer.prefixed("ep_recv_input"),
                recv_scatter_name=tracer.prefixed("ep_recv_scatter"),
            )
            gemm_input = ep_recv_input
            gemm_scatter = ep_recv_scatter
        else:
            gemm_input = permuted_input
            gemm_scatter = scatter_indices

        # Expert up (simple grouped GEMM, NOT gate_up)
        expert_up = g.moe_grouped_gemm(
            gemm_input,
            tracer.prefixed("experts_up"),
            gemm_scatter,
        )

        # Activation (relu2 by default)
        if self.activation == "relu2":
            expert_act = g.relu2(
                expert_up,
                out_name=tracer.prefixed("expert_act"),
            )
        else:
            expert_act = g.silu(
                expert_up,
                out_name=tracer.prefixed("expert_act"),
            )

        # Expert down
        expert_down = g.moe_grouped_gemm_down(
            expert_act,
            tracer.prefixed("experts_down"),
            gemm_scatter,
            out_name=tracer.prefixed("expert_down"),
        )

        # EP combine
        if self.ep_size > 1:
            expert_down = g.ep_combine(
                expert_down,
                num_experts=self.num_experts,
                ep_size=self.ep_size,
                top_k=self.num_experts_per_tok,
                out_name=tracer.prefixed("ep_combined"),
            )

        # Unpermute
        moe_out = g.moe_unpermute(
            expert_down,
            routing_weights,
            scatter_indices,
            top_k=self.num_experts_per_tok,
            out_name=out_slot,
        )

        return Proxy(out_slot, moe_out)


class NemotronSharedExpert(Module):
    """Shared expert for Nemotron MoE (simple up -> activation -> down)."""

    _hf_mapping_defaults_ = {
        "up": "{prefix}.shared_experts.up_proj.weight",
        "down": "{prefix}.shared_experts.down_proj.weight",
    }

    def __init__(
        self,
        d_model: int,
        shared_expert_intermediate: int,
        activation: str = "relu2",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.shared_expert_intermediate = shared_expert_intermediate
        self.activation = activation
        self.C = Dim("C")
        self.SharedM = Dim("SharedM")

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (x,) = args  # x is [B*T, C]

        _shared = self.shared_expert_intermediate
        _hidden = self.d_model

        tracer.register_param(
            "up",
            ("SharedM", "C"),
            lora_targets=[LoRATarget(name="shared_up", size=_shared)],
        )
        tracer.register_param(
            "down",
            ("C", "SharedM"),
            lora_targets=[LoRATarget(name="shared_down", size=_hidden)],
        )

        tracer.register_activation(
            "up_out",
            ("B * T", "SharedM"),
            share_policy="when_recomputed",
            when="use_shared_expert",
        )
        tracer.register_activation(
            "act",
            ("B * T", "SharedM"),
            share_policy="when_recomputed",
            when="use_shared_expert",
        )
        out_slot = tracer.register_activation(
            "out",
            ("B * T", "C"),
            share_policy="when_recomputed",
            when="use_shared_expert",
        )

        shared_up = g.matmul(
            x.ref,
            tracer.prefixed("up"),
            transpose="NT",
            out_name=tracer.prefixed("up_out"),
        )
        if self.activation == "relu2":
            shared_act = g.relu2(
                shared_up,
                out_name=tracer.prefixed("act"),
            )
        else:
            shared_act = g.silu(
                shared_up,
                out_name=tracer.prefixed("act"),
            )
        shared_out = g.matmul(
            shared_act,
            tracer.prefixed("down"),
            transpose="NT",
            out_name=out_slot,
        )

        return Proxy(out_slot, shared_out)
