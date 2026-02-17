"""Expert Parallelism (EP) primitives.

These ops route tokens across GPUs for distributed expert computation.
ep_dispatch sends tokens to expert-owning GPUs via all-to-all.
ep_combine reverses the all-to-all to collect expert outputs.
"""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import primitive


@primitive(impl="kernels.ep_dispatch")
def ep_dispatch(
    permuted_input: Tensor["TotalTokens", "C"],
    routing_indices: Tensor["BT", "K", "int32"],
    scatter_indices: Tensor["TotalTokens", "int32"],
    *,
    num_experts: int,
    ep_size: int,
    top_k: int,
) -> tuple[Tensor["RecvTokens", "C"], Tensor["RecvTokens", "int32"]]:
    """EP dispatch: route permuted tokens to expert-owning GPUs via all-to-all.

    After moe_permute, tokens are sorted by expert. ep_dispatch splits them
    by destination GPU (expert_idx // num_local_experts) and performs A2A.

    Returns (recv_tokens, recv_scatter_indices) for local expert computation.
    """
    ...


@primitive(impl="kernels.ep_combine")
def ep_combine(
    expert_output: Tensor["RecvTokens", "C"],
    *,
    num_experts: int,
    ep_size: int,
    top_k: int,
) -> Tensor["TotalTokens", "C"]:
    """EP combine: reverse all-to-all to collect expert outputs.

    After local expert computation, sends results back to source GPUs
    and reorders to match the original moe_permute output order.
    """
    ...
