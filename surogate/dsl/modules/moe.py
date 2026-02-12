"""MoE (Mixture of Experts) Modules.

Provides reusable MoE module definitions with HF weight mapping defaults.
Two variants are provided:

- ``MoEExpertsGated``: Experts with gated activation (gate_proj + up_proj fused).
  Used by Qwen3-MoE and similar architectures with SwiGLU expert MLPs.

- ``MoEExpertsSimple``: Experts with simple activation (up_proj only, no gate).
  Used by Nemotron-H MoE with relu2 activation.

Both store expert weights in batched format [num_experts, ...] for efficient
grouped GEMM computation. Expert offloading is supported via ``offload_group``
metadata which signals the C++ runtime to store expert weights on CPU and
stream them to GPU on demand.
"""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import module, forward, Param
from ..graph_builder import graph
from ..dim import Dim, B, T
from ..hf import stack_experts, transform


@module
class MoEExpertsGated:
    """MoE with gated expert activation (SwiGLU-style: gate+up fused).

    Expert architecture per-expert: gate_proj + up_proj (fused) -> SwiGLU -> down_proj.
    Used by Qwen3-MoE and similar models.

    HF weight layout:
        {prefix}.gate.weight                              -> router_weight
        {prefix}.experts.{expert}.gate_proj.weight  }
        {prefix}.experts.{expert}.up_proj.weight    } --> experts_gate_up (fused)
        {prefix}.experts.{expert}.down_proj.weight        -> experts_down
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
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Typed dimensions
        self.C = Dim("C")
        self.M = Dim("M")
        self.E = Dim("E")
        self.K = Dim("K")
        self.MUp = 2 * self.M  # gate + up fused

    # Router weight
    router_weight = Param(Tensor["E", "C"])

    # Expert weights (batched format: [num_experts, ...])
    # offload_group="moe_experts" signals to the runtime that these should
    # be stored on CPU and streamed to GPU on demand when offloading is enabled.
    experts_gate_up = Param(Tensor["E", "MUp", "C"], offload_group="moe_experts")
    experts_down = Param(Tensor["E", "C", "M"], offload_group="moe_experts")

    @forward
    def forward(self, x: Tensor["B * T", "C"]) -> Tensor["B * T", "C"]:
        """Placeholder forward - actual MoE computation is in block definitions."""
        with graph() as g:
            return x


@module
class MoEExpertsSimple:
    """MoE with simple expert activation (no gate, just up+down).

    Expert architecture per-expert: up_proj -> activation -> down_proj.
    Used by Nemotron-H MoE with relu2 activation.

    HF weight layout:
        {prefix}.gate.weight                              -> router_weight
        {prefix}.experts.{expert}.up_proj.weight          -> experts_gate_up (no fusion)
        {prefix}.experts.{expert}.down_proj.weight        -> experts_down

    Note: Uses ``experts_gate_up`` param name for compatibility with the grouped
    GEMM infrastructure, even though there is no gate projection. The param
    contains only the up_proj weights (fuse_gate_up=False).
    """

    _hf_mapping_defaults_ = {
        "router_weight": "{prefix}.gate.weight",
        "experts_gate_up": stack_experts(
            "{prefix}.experts.{expert}.up_proj.weight",
            fuse_gate_up=False,
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
        num_experts_per_tok: int = 2,
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Typed dimensions
        self.C = Dim("C")
        self.M = Dim("M")
        self.E = Dim("E")
        self.K = Dim("K")

    # Router weight
    router_weight = Param(Tensor["E", "C"])

    # Expert weights (batched format)
    experts_gate_up = Param(Tensor["E", "M", "C"], offload_group="moe_experts")
    experts_down = Param(Tensor["E", "C", "M"], offload_group="moe_experts")

    @forward
    def forward(self, x: Tensor["B * T", "C"]) -> Tensor["B * T", "C"]:
        """Placeholder forward - actual MoE computation is in block definitions."""
        with graph() as g:
            return x


@module
class GptOssMoE:
    """GPT-OSS MoE router + experts with per-expert biases.

    HF weight layout:
        {prefix}.router.weight          -> router_weight
        {prefix}.router.bias            -> router_bias
        {prefix}.experts.gate_up_proj   -> experts_gate_up (transpose)
        {prefix}.experts.gate_up_proj_bias -> experts_gate_up_bias
        {prefix}.experts.down_proj      -> experts_down (transpose)
        {prefix}.experts.down_proj_bias -> experts_down_bias
    """

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
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Typed dimensions
        self.C = Dim("C")
        self.M = Dim("M")
        self.E = Dim("E")
        self.K = Dim("K")
        self.MUp = 2 * self.M  # interleaved gate + up

    # Router weight + bias
    router_weight = Param(Tensor["E", "C"])
    router_bias = Param(Tensor["E"])

    # Expert weights/biases (batched format)
    experts_gate_up = Param(Tensor["E", "MUp", "C"], offload_group="moe_experts")
    experts_gate_up_bias = Param(Tensor["E", "MUp"], offload_group="moe_experts")
    experts_down = Param(Tensor["E", "C", "M"], offload_group="moe_experts")
    experts_down_bias = Param(Tensor["E", "C"], offload_group="moe_experts")

    @forward
    def forward(self, x: Tensor["B * T", "C"]) -> Tensor["B * T", "C"]:
        """Placeholder forward - actual MoE computation is in block definitions."""
        with graph() as g:
            return x


@module
class MoESharedExpert:
    """Shared expert weights for MoE models.

    Some MoE architectures (e.g., Qwen3-MoE) have a shared expert that
    processes all tokens in addition to the routed experts. The shared expert
    uses separate gate/up/down projections (SwiGLU-style).

    HF weight layout:
        {prefix}.shared_expert.gate_proj.weight -> shared_expert_gate
        {prefix}.shared_expert.up_proj.weight   -> shared_expert_up
        {prefix}.shared_expert.down_proj.weight -> shared_expert_down
    """

    _hf_mapping_defaults_ = {
        "shared_expert_gate": "{prefix}.shared_expert.gate_proj.weight",
        "shared_expert_up": "{prefix}.shared_expert.up_proj.weight",
        "shared_expert_down": "{prefix}.shared_expert.down_proj.weight",
    }

    def __init__(
        self,
        d_model: int,
        shared_expert_intermediate: int,
    ):
        self.d_model = d_model
        self.shared_expert_intermediate = shared_expert_intermediate

        # Typed dimensions
        self.C = Dim("C")
        self.SharedM = Dim("SharedM")

    # Shared expert weights
    shared_expert_gate = Param(Tensor["SharedM", "C"])
    shared_expert_up = Param(Tensor["SharedM", "C"])
    shared_expert_down = Param(Tensor["C", "SharedM"])

    @forward
    def forward(self, x: Tensor["B * T", "C"]) -> Tensor["B * T", "C"]:
        """Placeholder forward - actual computation is in block definitions."""
        with graph() as g:
            return x
