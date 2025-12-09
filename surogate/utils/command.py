import abc
from typing import Any

import datasets
import transformers

from surogate.core.config.loader import SurogateConfig
from surogate.utils.logger import get_logger
from surogate.utils.system_info import get_system_info
from surogate.utils.tensor import seed_everything

logger = get_logger()

from surogate.utils.dict import DictDefault


class SurogateCommand(abc.ABC):
    config: SurogateConfig
    model: Any
    
    def __init__(self, *, config: SurogateConfig, args: DictDefault):
        self.args = DictDefault(args)
        self.config = config
        self.system_info = get_system_info()

        transformers.logging.set_verbosity_error()
        datasets.logging.set_verbosity_error()

        if hasattr(self.config, 'seed') and self.config.seed:
            seed_everything(self.config.seed)
