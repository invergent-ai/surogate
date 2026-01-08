from typing import Dict, Optional, Any, List

from surogate.core.datasets.preprocessor.row import RowPreprocessor
from surogate.core.model.chat_templates.base import ChatTemplateProcessor


class EncodePreprocessor(RowPreprocessor):
    def __init__(self, template: 'ChatTemplateProcessor'):
        super().__init__()
        self.template = template

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.template.encode(row, return_length=True)

    def preprocess_batch(self, rows: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """Process a batch of rows using batched tokenization.

        This is significantly faster than processing rows individually because:
        1. Tokenizer can process multiple texts in parallel
        2. Reduces Python function call overhead
        3. Better CPU cache utilization

        Args:
            rows: List of input row dictionaries

        Returns:
            List of encoded dictionaries (or None for failed rows)
        """
        return self.template.encode_batch(rows, return_length=True)
