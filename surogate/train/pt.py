from surogate.core.config.sft_config import SFTConfig
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger
from surogate.train.sft import SurogateSFT
import datasets

datasets.logging.set_verbosity_warning()

logger = get_logger()

class SurogatePT(SurogateSFT):
    def __init__(self, config: SFTConfig, args: DictDefault):
        config.use_chat_template = False
        config.loss_scale = 'all'
        config.init_projections_to_zero = True
        config.from_scratch = True
        super().__init__(config=config, args=args)
        
        
def pt_main(config: SFTConfig, args: DictDefault):
    SurogatePT(config, args).run()