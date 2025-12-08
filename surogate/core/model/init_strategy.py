import torch
from torch import nn

from surogate.utils.logger import get_logger

logger = get_logger()


class InitModelStrategy:
    @staticmethod
    def constant_init(param: torch.Tensor, c: float = 0) -> None:
        nn.init.constant_(param, c)

    @staticmethod
    def uniform_init(param: torch.Tensor, a: float = -0.1, b: float = 0.1) -> None:
        nn.init.uniform_(param, a, b)

    @staticmethod
    def normal_init(param: torch.Tensor, mean: float = 0.0, std: float = 0.01) -> None:
        nn.init.normal_(param, mean, std)

    @staticmethod
    def is_uninitialized(param: torch.Tensor) -> bool:
        """
        Check if a parameter is uninitialized or has numerically unstable values.
        Criteria:
            - Tensor has NaN or Inf values
            - Tensor stats (mean or std) are outside reasonable range
        """
        if param.numel() == 0:
            return False

        with torch.no_grad():
            mean_abs = param.abs().mean()
            std = param.std()

            # NaN or Inf
            if not torch.isfinite(mean_abs) or not torch.isfinite(std):
                return True

            # Use empirically safe threshold
            MAX_THRESHOLD = 1e7
            if mean_abs > MAX_THRESHOLD or std > MAX_THRESHOLD:
                return True

            return False

    _INIT_STRATEGY_MAP = {
        'zero': constant_init,
        'uniform': uniform_init,
        'normal': normal_init
    }

    @staticmethod
    def init_parameters(model: nn.Module, init_strategy: str) -> None:
        """
        Initialize model parameters using the specified strategy.

        Args:
            model: The model whose parameters to initialize
            init_strategy: Name of initialization strategy
        """
        if init_strategy not in InitModelStrategy._INIT_STRATEGY_MAP:
            raise ValueError(f'Unknown initialization strategy: {init_strategy}')

        logger.info(f'initialization strategy: {init_strategy}')

        init_func = InitModelStrategy._INIT_STRATEGY_MAP[init_strategy]

        for name, param in model.named_parameters():
            if InitModelStrategy.is_uninitialized(param):
                logger.info(f'Initializing parameters: {name}.')
                init_func(param)
