from typing import Dict, Any, Optional

from swift.llm.dataset import RowPreprocessor

from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger
from surogate.utils.schema.datasets import TextDataset

logger = get_logger()

class TextPreprocessor(RowPreprocessor):
    def __init__(self, cfg: DictDefault, dataset_config: TextDataset):
        super().__init__()
        self.cfg = cfg
        self.ds_cfg = dataset_config

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text = row.pop(self.ds_cfg.text_field, "")
        if text == "":
            logger.warning("Found empty value in text field. Please check your dataset.")

        messages = [{
            'role': 'user',
            'content': text
        }]
        row.update({'messages': messages})
        return row
