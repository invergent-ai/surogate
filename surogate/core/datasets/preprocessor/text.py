from typing import Dict, Any, Optional, List

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

    def preprocess_batch(self, rows: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """Process a batch of text dataset rows efficiently.

        1. Batches field extraction operations
        2. Reduces function call overhead
        3. Batches message creation

        Args:
            rows: List of input row dictionaries

        Returns:
            List of processed dictionaries with 'messages' field
        """
        results = []

        for row in rows:
            # Extract text field (use .get() for efficiency)
            text = row.get(self.ds_cfg.text_field, "")
            if text == "":
                logger.warning("Found empty value in text field. Please check your dataset.")

            # Create message
            messages = [{
                'role': 'user',
                'content': text
            }]

            # Preserve remaining fields
            result = {'messages': messages}
            result.update({k: v for k, v in row.items() if k != self.ds_cfg.text_field})
            results.append(result)

        return results
