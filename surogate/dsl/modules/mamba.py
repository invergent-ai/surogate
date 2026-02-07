"""Mamba2 Modules for State Space Model based sequence mixing."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import module, forward, Param
from ..graph_builder import graph
from ..dim import Dim, B, T


@module
class Mamba2Mixer:
    """Mamba2 mixer module for hybrid architectures like Nemotron-H.

    Implements the Mamba2 State Space Model with:
    - Input projection (produces gate, hidden_states for conv, B, C, dt)
    - Causal 1D convolution
    - Selective state space scan (SSD)
    - Gated RMSNorm
    - Output projection

    This is based on the Mamba2 paper and Nemotron-H implementation.
    """

    # Default HF weight path templates.
    # Use {prefix} for the Mamba mixer submodule path
    # (e.g., "backbone.layers.{layer}.mixer" for Nemotron-H).
    _hf_mapping_defaults_ = {
        "in_proj_weight": "{prefix}.in_proj.weight",
        "in_proj_bias": "{prefix}.in_proj.bias",
        "conv_weight": "{prefix}.conv1d.weight",
        "conv_bias": "{prefix}.conv1d.bias",
        "A_log": "{prefix}.A_log",
        "D_param": "{prefix}.D",
        "dt_bias": "{prefix}.dt_bias",
        "gated_norm_weight": "{prefix}.norm.weight",
        "out_proj_weight": "{prefix}.out_proj.weight",
        "out_proj_bias": "{prefix}.out_proj.bias",
    }

    def __init__(
        self,
        d_model: int,
        mamba_num_heads: int = 128,
        mamba_head_dim: int = 64,
        ssm_state_size: int = 128,
        n_groups: int = 8,
        conv_kernel: int = 4,
        chunk_size: int = 256,
        eps: float = 1e-5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        use_conv_bias: bool = True,
        use_bias: bool = False,
    ):
        self.d_model = d_model
        self.mamba_num_heads = mamba_num_heads
        self.mamba_head_dim = mamba_head_dim
        self.ssm_state_size = ssm_state_size
        self.n_groups = n_groups
        self.conv_kernel = conv_kernel
        self.chunk_size = chunk_size
        self.eps = eps
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.use_conv_bias = use_conv_bias
        self.use_bias = use_bias

        # Derived dimensions
        self.intermediate_size = mamba_num_heads * mamba_head_dim
        self.conv_dim = self.intermediate_size + 2 * n_groups * ssm_state_size
        self.projection_size = self.intermediate_size + self.conv_dim + mamba_num_heads

        # Typed dimensions
        self.C = Dim("C")           # d_model
        self.I = Dim("I")           # intermediate_size
        self.H = Dim("H")           # mamba_num_heads
        self.D = Dim("D")           # mamba_head_dim
        self.N = Dim("N")           # ssm_state_size
        self.G = Dim("G")           # n_groups
        self.K = Dim("K")           # conv_kernel
        self.P = Dim("P")           # projection_size
        self.D_conv = Dim("D_conv") # conv_dim

    # Input projection
    in_proj_weight = Param(Tensor["P", "C"])
    in_proj_bias = Param(Tensor["P"], when="use_bias")

    # Convolution
    conv_weight = Param(Tensor["D_conv", "K"])
    conv_bias = Param(Tensor["D_conv"], when="use_conv_bias")

    # SSM parameters
    A_log = Param(Tensor["H"])            # Log of state decay (negative for stability)
    D_param = Param(Tensor["H"])          # Skip connection weight
    dt_bias = Param(Tensor["H"])          # Time step bias

    # Gated RMSNorm
    gated_norm_weight = Param(Tensor["I"], quantizable=False)

    # Output projection
    out_proj_weight = Param(Tensor["C", "I"])
    out_proj_bias = Param(Tensor["C"], when="use_bias")

    @forward
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
        """Forward pass through Mamba2 mixer.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        with graph() as g:
            # Input projection
            x_flat = g.view(x, shape=[B * T, self.C])
            if self.use_bias:
                projected_flat = g.matmul_bias(x_flat, "in_proj_weight", "in_proj_bias", transpose="NT")
            else:
                projected_flat = g.matmul(x_flat, "in_proj_weight", transpose="NT")
            projected = g.view(projected_flat, shape=[B, T, self.P])

            # Split projection into gate, conv_input, dt
            gate, conv_input, dt = g.mamba_split_proj(
                projected,
                intermediate_size=self.intermediate_size,
                conv_dim=self.conv_dim,
                num_heads=self.mamba_num_heads,
            )

            # Causal 1D convolution
            if self.use_conv_bias:
                conv_out = g.mamba_conv1d(conv_input, "conv_weight", "conv_bias", activation="silu")
            else:
                conv_out = g.mamba_conv1d(conv_input, "conv_weight", None, activation="silu")

            # Split conv output into hidden_states, B, C for SSM
            hidden_states, ssm_B, ssm_C = g.mamba_split_conv_out(
                conv_out,
                intermediate_size=self.intermediate_size,
                groups_state_size=self.n_groups * self.ssm_state_size,
                n_groups=self.n_groups,
                ssm_state_size=self.ssm_state_size,
            )

            # Reshape for SSM scan
            hidden_states_4d = g.view(hidden_states, shape=[B, T, self.H, self.D])
            ssm_B_4d = g.view(ssm_B, shape=[B, T, self.G, self.N])
            ssm_C_4d = g.view(ssm_C, shape=[B, T, self.G, self.N])

            # State Space Model scan
            ssm_out, ssm_state = g.mamba_ssm_scan(
                hidden_states_4d, dt, "A_log", ssm_B_4d, ssm_C_4d, "D_param",
                dt_bias="dt_bias",
                dt_softplus=True,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                chunk_size=self.chunk_size,
                num_heads=self.mamba_num_heads,
                head_dim=self.mamba_head_dim,
                ssm_state_size=self.ssm_state_size,
                n_groups=self.n_groups,
            )

            # Reshape SSM output
            ssm_out_flat = g.view(ssm_out, shape=[B, T, self.I])

            # Gated RMSNorm
            gated_out = g.mamba_gated_rmsnorm(
                ssm_out_flat, gate, "gated_norm_weight",
                eps=self.eps,
                group_size=self.intermediate_size // self.n_groups,
            )

            # Output projection
            gated_flat = g.view(gated_out, shape=[B * T, self.I])
            if self.use_bias:
                out_flat = g.matmul_bias(gated_flat, "out_proj_weight", "out_proj_bias", transpose="NT")
            else:
                out_flat = g.matmul(gated_flat, "out_proj_weight", transpose="NT")
            out = g.view(out_flat, shape=[B, T, self.C])

            return out


@module
class SimpleMLP:
    """Simple MLP with relu2 activation (used in Nemotron-H).

    Architecture: up -> relu2 -> down
    """

    # Default HF weight path templates.
    # Use {prefix} for the MLP submodule path
    # (e.g., "backbone.layers.{layer}.mixer" for Nemotron-H).
    _hf_mapping_defaults_ = {
        "up_weight": "{prefix}.up_proj.weight",
        "up_bias": "{prefix}.up_proj.bias",
        "down_weight": "{prefix}.down_proj.weight",
        "down_bias": "{prefix}.down_proj.bias",
    }

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "relu2",
        use_bias: bool = False,
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.use_bias = use_bias

        # Typed dimensions
        self.C = Dim("C")
        self.M = Dim("M")

    # MLP weights
    up_weight = Param(Tensor["M", "C"])
    up_bias = Param(Tensor["M"], when="use_bias")
    down_weight = Param(Tensor["C", "M"])
    down_bias = Param(Tensor["C"], when="use_bias")

    @forward
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
        """Forward pass.

        Args:
            x: Input [B, T, C]

        Returns:
            Output [B, T, C]
        """
        with graph() as g:
            x_flat = g.view(x, shape=[B * T, self.C])

            # Up projection
            if self.use_bias:
                up_flat = g.matmul_bias(x_flat, "up_weight", "up_bias", transpose="NT")
            else:
                up_flat = g.matmul(x_flat, "up_weight", transpose="NT")

            # Activation
            if self.activation == "relu2":
                act_flat = g.relu2(up_flat)
            elif self.activation == "silu":
                act_flat = g.silu(up_flat)
            elif self.activation == "gelu":
                act_flat = g.gelu(up_flat)
            else:
                act_flat = g.relu2(up_flat)

            # Down projection
            if self.use_bias:
                out_flat = g.matmul_bias(act_flat, "down_weight", "down_bias", transpose="NT")
            else:
                out_flat = g.matmul(act_flat, "down_weight", transpose="NT")

            out = g.view(out_flat, shape=[B, T, self.C])
            return out
