from typing import Any, Dict, Optional, List

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
        instruction = row.get('instruction', None)
        input_ = row.get('input', None)
        output = row.get('output', None)
        if output is not None:
            row['response'] = output
        row['query'] = self.concat_inst_input(instruction, input_)
        return super().preprocess(row)

    def preprocess_batch(self, rows: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """Process a batch of Alpaca dataset rows efficiently.

        1. Batches instruction/input concatenation operations
        2. Reduces function call overhead
        3. Batches query/response extraction

        Args:
            rows: List of input row dictionaries

        Returns:
            List of processed dictionaries with 'query' and 'response' fields
        """
        # Extract and concatenate instructions/inputs in batch
        processed_rows = []
        for row in rows:
            instruction = row.get('instruction')
            input_ = row.get('input')
            output = row.get('output')

            new_row = {k: v for k, v in row.items() if k not in {'instruction', 'input', 'output'}}

            if output is not None:
                new_row['response'] = output
            new_row['query'] = self.concat_inst_input(instruction, input_)
            processed_rows.append(new_row)

        # Call parent's preprocess_batch if it exists, otherwise fallback to per-row
        if hasattr(super(), 'preprocess_batch'):
            return super().preprocess_batch(processed_rows)
        else:
            # Fallback to sequential processing via parent preprocess
            return [super().preprocess(row) for row in processed_rows]
