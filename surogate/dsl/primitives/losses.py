"""Loss primitives."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import primitive, save


@primitive(impl="kernels.fused_lm_head_loss")
def fused_lm_head_loss(
    xF_flat: Tensor["B*T", "d_model"],
    weight: Tensor["vocab_size", "d_model"],
    targets: Tensor["B*T", "int32"],
    *,
    compute_accuracy: bool = False,
) -> Tensor["B*T", "fp32"]:
    """Fused LM head matmul + cross-entropy loss.

    Computes per-token loss without materializing full logits for large vocabularies.
    """
    ...


@fused_lm_head_loss.backward
@save("xF_flat", "weight")
def fused_lm_head_loss_backward(
    d_loss: Tensor["B*T", "fp32"],
    xF_flat: Tensor["B*T", "d_model"],
    weight: Tensor["vocab_size", "d_model"],
    targets: Tensor["B*T", "int32"],
) -> tuple[Tensor["B*T", "d_model"], Tensor["vocab_size", "d_model"]]:
    """Backward pass for fused LM head + loss."""
    ...
