from typing import Dict, Any, Optional

from surogate.core.config.dataset_config import TextDatasetConfig
from surogate.core.datasets.preprocessor.row import RowPreprocessor
from surogate.utils.logger import get_logger

logger = get_logger()

class TextPreprocessor(RowPreprocessor):
    def __init__(self, dataset_config: TextDatasetConfig):
        super().__init__()
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
