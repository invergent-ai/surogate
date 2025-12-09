from typing import Any, Dict, Optional

from surogate.core.datasets.preprocessor.response import ResponsePreprocessor


class AlpacaPreprocessor(ResponsePreprocessor):

    @classmethod
    def concat_inst_input(cls, instruction, input_):
        if instruction and input_:
            query = f'{instruction}\n{input_}'
        else:
            query = instruction or input_
        assert isinstance(query, str), f'query: {query}'
        return query

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        instruction = row.pop('instruction', None)
        input_ = row.pop('input', None)
        output = row.pop('output', None)
        if output is not None:
            row['response'] = output
        row['query'] = self.concat_inst_input(instruction, input_)
        return super().preprocess(row)
