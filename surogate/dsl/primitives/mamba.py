"""Mamba2 / State Space Model primitives for hybrid architectures like Nemotron-H."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import primitive, save


@primitive(impl="kernels.mamba_conv1d")
def mamba_conv1d(
    x: Tensor["B", "D_conv", "T"],
    weight: Tensor["D_conv", "K"],
    bias: Tensor["D_conv"] | None = None,
    *,
    activation: str = "silu",
) -> Tensor["B", "D_conv", "T"]:
    """Causal 1D convolution for Mamba.

    Args:
        x: Input tensor [B, D_conv, T] where D_conv = intermediate_size + 2 * n_groups * ssm_state_size
        weight: Convolution weight [D_conv, kernel_size]
        bias: Optional bias [D_conv]
        activation: Activation function ("silu" or "swish")

    Returns:
        Convolved output [B, D_conv, T]
    """
    ...


@mamba_conv1d.backward
@save("x")
def mamba_conv1d_backward(
    d_out: Tensor["B", "D_conv", "T"],
    x: Tensor["B", "D_conv", "T"],
    weight: Tensor["D_conv", "K"],
) -> tuple[Tensor["B", "D_conv", "T"], Tensor["D_conv", "K"], Tensor["D_conv"] | None]:
    """Backward pass for causal 1D convolution."""
    ...


@primitive(impl="kernels.mamba_ssm_scan")
def mamba_ssm_scan(
    hidden_states: Tensor["B", "I", "T"],
    dt: Tensor["B", "I", "T"],
    A: Tensor["H"],
    B: Tensor["B", "G", "N", "T"],
    C: Tensor["B", "G", "N", "T"],
    D: Tensor["H"],
    *,
    dt_bias: Tensor["H"] | None = None,
    dt_softplus: bool = True,
    dt_min: float = 0.0,
    dt_max: float = 1e9,
    chunk_size: int = 256,
) -> tuple[Tensor["B", "T", "I"], Tensor["B", "H", "D", "N"]]:
    """Mamba2 State Space Model scan (SSD algorithm).

    Implements the selective state space model:
        h_{t+1} = exp(dt * A) * h_t + dt * B_t * x_t
        y_t = C_t * h_t + D * x_t

    Args:
        hidden_states: Input hidden states [B, I, T] (channel-first, I = num_heads * head_dim)
        dt: Time step / delta [B, I, T] (expanded per-head)
        A: State decay (typically negative log of learned parameter) [num_heads]
        B: Input-to-state projection [B, n_groups, ssm_state_size, T]
        C: State-to-output projection [B, n_groups, ssm_state_size, T]
        D: Skip connection weight [num_heads]
        dt_bias: Bias for dt (applied before softplus)
        dt_softplus: Whether to apply softplus to dt
        dt_min: Minimum value for dt after softplus
        dt_max: Maximum value for dt after softplus
        chunk_size: Chunk size for chunked scan algorithm

    Returns:
        (output, final_state):
            output: SSM output [B, T, I]
            final_state: Final SSM state [B, num_heads, head_dim, ssm_state_size]
    """
    ...


@mamba_ssm_scan.backward
@save("hidden_states", "dt", "B", "C", "final_state")
def mamba_ssm_scan_backward(
    d_out: Tensor["B", "T", "I"],
    d_final_state: Tensor["B", "H", "D", "N"] | None,
    hidden_states: Tensor["B", "I", "T"],
    dt: Tensor["B", "I", "T"],
    A: Tensor["H"],
    B: Tensor["B", "G", "N", "T"],
    C: Tensor["B", "G", "N", "T"],
    D: Tensor["H"],
    final_state: Tensor["B", "H", "D", "N"],
) -> tuple[
    Tensor["B", "I", "T"],       # d_hidden_states
    Tensor["B", "I", "T"],       # d_dt
    Tensor["H"],                  # d_A
    Tensor["B", "G", "N", "T"],  # d_B
    Tensor["B", "G", "N", "T"],  # d_C
    Tensor["H"],                  # d_D
]:
    """Backward pass for SSM scan."""
    ...


@primitive(impl="kernels.mamba_gated_rmsnorm")
def mamba_gated_rmsnorm(
    x: Tensor["*", "C"],
    gate: Tensor["*", "C"],
    weight: Tensor["C"],
    *,
    eps: float = 1e-5,
    group_size: int = 0,
    norm_before_gate: bool = False,
) -> Tensor["*", "C"]:
    """Gated RMSNorm for Mamba2.

    Applies RMSNorm with a multiplicative gate:
        if norm_before_gate:
            out = RMSNorm(x) * gate
        else:
            out = RMSNorm(x * gate)  # Mamba default

    Args:
        x: Input tensor
        gate: Gate tensor (same shape as x)
        weight: RMSNorm weight
        eps: Epsilon for numerical stability
        group_size: Group size for group normalization (0 for full dim)
        norm_before_gate: If True, normalize before gating

    Returns:
        Gated normalized output
    """
    ...


@mamba_gated_rmsnorm.backward
@save("x", "gate")
def mamba_gated_rmsnorm_backward(
    d_out: Tensor["*", "C"],
    x: Tensor["*", "C"],
    gate: Tensor["*", "C"],
    weight: Tensor["C"],
) -> tuple[Tensor["*", "C"], Tensor["*", "C"], Tensor["C"]]:
    """Backward pass for gated RMSNorm. Returns (d_x, d_gate, d_weight)."""
    ...


@primitive(impl="kernels.mamba_split_proj")
def mamba_split_proj(
    projected: Tensor["B", "T", "P"],
    *,
    intermediate_size: int,
    conv_dim: int,
    num_heads: int,
    head_dim: int,
) -> tuple[
    Tensor["B", "T", "I"],      # gate
    Tensor["B", "D_conv", "T"], # hidden_states_B_C (for conv)
    Tensor["B", "I", "T"],      # dt (expanded)
]:
    """Split Mamba2 input projection into components.

    The input projection produces [d_mlp, d_mlp, gate, hidden_states_B_C, dt]
    but d_mlp parts are not used in standard Mamba2. This splits out:
    - gate: for gating the output
    - hidden_states_B_C: input to conv + SSM (channel-first)
    - dt: time step deltas (expanded from per-head to per-element via head_dim, channel-first)

    Args:
        projected: Full input projection output [B, T, projection_size]
        intermediate_size: Size of the gate (and hidden states before conv)
        conv_dim: Size of conv input (intermediate_size + 2 * n_groups * ssm_state_size)
        num_heads: Number of heads (size of dt in projection)
        head_dim: Dimension per head (dt is expanded from num_heads to num_heads * head_dim)

    Returns:
        (gate, hidden_states_B_C, dt)
    """
    ...


@primitive(impl="kernels.mamba_split_conv_out")
def mamba_split_conv_out(
    conv_out: Tensor["B", "D_conv", "T"],
    *,
    intermediate_size: int,
    groups_state_size: int,
) -> tuple[
    Tensor["B", "I", "T"],  # hidden_states
    Tensor["B", "G", "N", "T"], # B (input-to-state)
    Tensor["B", "G", "N", "T"], # C (state-to-output)
]:
    """Split convolution output into hidden states, B, and C.

    Args:
        conv_out: Convolution output [B, conv_dim, T]
        intermediate_size: Size of hidden states
        groups_state_size: n_groups * ssm_state_size

    Returns:
        (hidden_states, B, C)
    """
    ...


@primitive(impl="kernels.mamba_combine_scan")
def mamba_combine_scan(
    projected_states: Tensor["B", "T", "P"],
    conv_weight: Tensor["D_conv", "K"],
    conv_bias: Tensor["D_conv"] | None,
    dt_bias: Tensor["H"],
    A_log: Tensor["H"],
    D: Tensor["H"],
    norm_weight: Tensor["I"],
    out_proj_weight: Tensor["C", "I"],
    out_proj_bias: Tensor["C"] | None,
    *,
    chunk_size: int = 256,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    intermediate_size: int,
    ssm_state_size: int,
    eps: float = 1e-5,
    activation: str = "silu",
    dt_min: float = 0.0,
    dt_max: float = 1e9,
) -> Tensor["B", "T", "C"]:
    """Fused Mamba2 operation: conv1d + SSM scan + gated norm + output proj.

    This is the fused forward path that combines all Mamba2 operations
    for maximum efficiency during training.

    Args:
        projected_states: Input projection output
        conv_weight: Conv1d weight
        conv_bias: Conv1d bias
        dt_bias: Time step bias
        A_log: Log of state decay parameter
        D: Skip connection weight
        norm_weight: Gated RMSNorm weight
        out_proj_weight: Output projection weight
        out_proj_bias: Output projection bias
        chunk_size: Chunk size for SSM scan
        num_heads: Number of Mamba heads
        head_dim: Dimension per head
        n_groups: Number of groups for B, C
        intermediate_size: num_heads * head_dim
        ssm_state_size: State dimension
        eps: RMSNorm epsilon
        activation: Activation for conv ("silu")
        dt_min, dt_max: Time step clipping bounds

    Returns:
        Output tensor [B, T, hidden_size]
    """
    ...
