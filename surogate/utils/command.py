from surogate.utils.config import load_config
from abc import ABC, abstractmethod

from surogate.utils.dict import DictDefault


class SurogateCommand(ABC):
    def __init__(self, **kwargs):
        self.args = DictDefault(kwargs)
        self.config = load_config(self.args['config'])

    @abstractmethod
    def run(self):
        pass
