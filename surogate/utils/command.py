import abc
import logging
from typing import Any

from surogate.utils.logger import get_logger
from surogate.utils.system_info import get_system_info
from surogate.utils.tensor import seed_everything

logger = get_logger()

from surogate.utils.dict import DictDefault


class SurogateCommand(abc.ABC):
    config: Any
    model: Any

    def __init__(self, *, config, args: DictDefault):
        self.args = DictDefault(args)
        self.config = config
        self.system_info = get_system_info()

        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.ERROR)

        if hasattr(self.config, "seed") and self.config.seed:
            seed_everything(self.config.seed)
