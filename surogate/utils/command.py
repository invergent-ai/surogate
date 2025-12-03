import abc
import os
from typing import Any

import datasets
import transformers

from surogate.config.loader import SurogateConfig
from surogate.utils.logger import get_logger
from surogate.utils.system_info import get_system_info
from swift.llm import BaseArguments
from swift.utils import seed_everything, get_dist_setting

logger = get_logger()

from surogate.utils.dict import DictDefault


class SurogateCommand(abc.ABC):
    config: SurogateConfig
    model: Any
    
    def __init__(self, *, config: SurogateConfig, args: DictDefault, swift=False):
        self.sg_args = DictDefault(args)
        self.sg_config = config
        self.system_info = get_system_info()

        transformers.logging.set_verbosity_error()
        datasets.logging.set_verbosity_error()

        if swift:
            super().__init__(self.to_swift_args())
        else:
            if hasattr(self.sg_config, 'seed') and self.sg_config.seed:
                seed_everything(self.sg_config.seed)

        self.print_distributed_config()

    def to_swift_args(self) -> BaseArguments:
        pass

    def print_distributed_config(self):
        rank, local_rank, world_size, local_world_size = get_dist_setting()
        is_distributed = world_size > 1
        logger.info(f"Distributed Environment: {'Yes' if is_distributed else 'No'}")
        logger.info(f"World Size: {world_size}, Local Rank: {local_rank}")
