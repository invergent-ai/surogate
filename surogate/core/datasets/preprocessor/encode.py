from typing import Dict, Optional, Any

from surogate.core.datasets.preprocessor.row import RowPreprocessor
from surogate.core.model.chat_templates.base import ChatTemplateProcessor


class EncodePreprocessor(RowPreprocessor):
    def __init__(self, template: 'ChatTemplateProcessor'):
        super().__init__()
        self.template = template

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.template.encode(row, return_length=True)

class AddLengthPreprocessor(EncodePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        encoded = super().preprocess(row)
        row['length'] = encoded['length']
        return row