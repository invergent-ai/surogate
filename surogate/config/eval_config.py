from dataclasses import dataclass

from surogate.utils.dict import DictDefault


@dataclass
class EvalConfig:
    def __init__(self, cfg: DictDefault):
        # TODO: set props from cfg
        self.__post_init__()

    def __post_init__(self):
        # TODO: validate props
        pass
