from surogate.utils.config import load_config
from abc import ABC, abstractmethod

class SurogateCommand(ABC):
    def __init__(self, **kwargs):
        self.args = kwargs
        self.config = load_config(self.args['config'])

    @abstractmethod
    def run(self):
        pass