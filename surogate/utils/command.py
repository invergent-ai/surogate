import abc
from typing import Any

from swift.llm import BaseArguments, TrainArguments
from swift.utils import seed_everything

from surogate.config.loader import SurogateConfig
from surogate.utils.logger import get_logger

logger = get_logger()

from surogate.utils.dict import DictDefault


class SurogateCommand(abc.ABC):
    config: SurogateConfig
    model: Any
    
    def __init__(self, *, config: SurogateConfig, args: DictDefault, swift=False):
        self.sg_args = DictDefault(args)
        self.sg_config = config

        if swift:
            super().__init__(self.to_swift_args())
        else:
            if hasattr(self.sg_config, 'seed') and self.sg_config.seed:
                seed_everything(self.sg_config.seed)

    def to_swift_args(self) -> BaseArguments:
        pass
