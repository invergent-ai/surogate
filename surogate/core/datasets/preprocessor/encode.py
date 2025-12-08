from typing import Dict, Optional, Any

from surogate.core.datasets.preprocessor.row import RowPreprocessor
from surogate.core.model.chat_templates.base import ChatTemplateProcessor


class EncodePreprocessor(RowPreprocessor):
    def __init__(self, template: 'ChatTemplateProcessor', pre_tokenize: bool = False):
        super().__init__()
        self.template = template
        self.pre_tokenize = pre_tokenize

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        encoded = self.template.encode(row, return_length=True)
        if self.pre_tokenize:
            row['length'] = encoded['length']
            encoded = row
        return encoded