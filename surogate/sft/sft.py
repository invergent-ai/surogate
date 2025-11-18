from surogate.config.sft_config import SFTConfig
from surogate.utils.command import SurogateCommand


class SurogateSFT(SurogateCommand):
    config: SFTConfig

    def __init__(self, **kwargs):
        super().__init__(SFTConfig, **kwargs)

    def run(self):
        pass