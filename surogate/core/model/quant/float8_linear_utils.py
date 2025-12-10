from typing import Optional, Callable

from torch import nn
from torchao.float8 import Float8LinearConfig
from torchao.float8.float8_linear_utils import swap_linear_layers

from surogate.core.model.quant.float8_linear_falqon import Float8Linear_falqon


def convert_to_float8_training_falqon(
        module: nn.Module,
        *,
        module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
        config: Float8LinearConfig = None,
        rank: Optional[int] = 32,
        lora_alpha: Optional[float] = 16,
        lora_init: Optional[str] = "svd",
        num_topk: Optional[int] = 10,
) -> nn.Module:
    """
    Swaps `torch.nn.Linear` in `module` with `Float8Linear`.

    Args:
        module: Module to modify.
        module_filter_fn: If specified, only the `torch.nn.Linear` subclasses that
            that pass the filter function will be swapped. The inputs to the
            filter function are the module instance and the FQN.
        config (Float8LinearConfig): configuration for conversion to float8

    Returns:
     nn.Module: The modified module with swapped linear layers.
    """
    if config is None:
        config = Float8LinearConfig()

    from_float = lambda m: Float8Linear_falqon.from_float(
        m,
        config=config,
        rank=rank,
        lora_alpha=lora_alpha,
        lora_init=lora_init,
        num_topk=num_topk,
    )
    return swap_linear_layers(
        module,
        from_float,
        module_filter_fn=module_filter_fn,
    )