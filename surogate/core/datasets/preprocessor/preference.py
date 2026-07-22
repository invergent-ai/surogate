from typing import Any

from surogate.core.config.dataset_config import PreferenceDatasetConfig
from surogate.core.datasets.preprocessor.row import RowPreprocessor
from surogate.utils.logger import get_logger

logger = get_logger()


class PreferencePreprocessor(RowPreprocessor):
    """Normalize preference-pair rows for offline DPO.

    Each output row is `{prompt, chosen, rejected}` plus the optional
    `enable_thinking` flag. `prompt` is a string or a chat messages list;
    `chosen`/`rejected` are the two competing assistant continuations
    (see surogate/dpo/data.py for the downstream tokenization). All other
    columns are dropped so every shard yields a uniform schema.
    """

    def __init__(self, dataset_config: PreferenceDatasetConfig):
        super().__init__()
        self.ds_cfg = dataset_config
        self.columns[dataset_config.prompt_field] = "prompt"
        self.columns[dataset_config.chosen_field] = "chosen"
        self.columns[dataset_config.rejected_field] = "rejected"
        self.columns[dataset_config.enable_thinking_field] = "enable_thinking"

    def preprocess(self, row: dict[str, Any]) -> dict[str, Any] | None:
        prompt = row.get("prompt")
        if not isinstance(prompt, (str, list)) or len(prompt) == 0:
            raise ValueError(f"'prompt' must be a non-empty string or messages list. Row: {row}")

        result = {"prompt": prompt}
        for key in ("chosen", "rejected"):
            response = row.get(key)
            if not isinstance(response, str) or response == "":
                raise ValueError(f"'{key}' must be a non-empty string. Row: {row}")
            result[key] = response

        enable_thinking = row.get("enable_thinking")
        if enable_thinking is not None:
            result["enable_thinking"] = bool(enable_thinking)
        return result

    def preprocess_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        """Process a batch of preference rows.

        Raises on the first invalid row; RowPreprocessor then falls back to
        row-by-row processing, which drops only the offending rows.
        """
        return [self.preprocess(row) for row in rows]
