import ast
import os
from typing import Any, Dict, Optional

from surogate.core.datasets.preprocessor.row import RowPreprocessor
from surogate.core.model.chat_templates.utils import history_to_messages


class ResponsePreprocessor(RowPreprocessor):
    def __init__(self, *, columns: Optional[Dict[str, str]] = None, **kwargs) -> None:
        super().__init__(columns=columns, **kwargs)
        system_keys = ['system', 'system_prompt']
        query_keys = ['query', 'prompt', 'input', 'instruction', 'question', 'problem']
        response_keys = ['response', 'answer', 'output', 'targets', 'target', 'answer_key', 'answers', 'solution'
                         ] + ['text', 'completion', 'content']
        for key in system_keys:
            self.columns[key] = 'system'
        for key in query_keys:
            self.columns[key] = 'query'
        for key in response_keys:
            self.columns[key] = 'response'

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        response = row.pop('response', None)
        if response is not None:
            if isinstance(response, (list, tuple)):
                from transformers.utils import strtobool
                # sometimes response is a list, pick one randomly
                if strtobool(os.environ.get('RANDOM_DATASET_RESPONSE', 'False')):
                    response = self.random_state.choice(response)
                else:
                    response = response[0]
        history = row.pop('history', None) or []
        query = row.pop('query', None)
        system = row.pop('system', None)
        if isinstance(history, str):  # e.g. "[['query1', 'response1']]"
            history = ast.literal_eval(history)
        history.append([query, response])

        row.update({'messages': history_to_messages(history, system)})
        return row