from surogate.core.datasets.preprocessor.alpaca import AlpacaPreprocessor
from surogate.core.datasets.preprocessor.messages import MessagesPreprocessor
from surogate.core.datasets.preprocessor.response import ResponsePreprocessor
from surogate.core.datasets.preprocessor.row import RowPreprocessor
from surogate.core.datasets.utils import DATASET_TYPE


class AutoPreprocessor:
    def __init__(self, *, columns: dict[str, str] | None = None, **kwargs) -> None:
        self.columns = columns or {}
        self.kwargs = kwargs

    def _get_preprocessor(self, dataset: DATASET_TYPE) -> RowPreprocessor:
        features = dataset.features
        for key in ["conversation", "conversations", "messages"]:
            if key in features:
                return MessagesPreprocessor(**self.kwargs)
        if "instruction" in features and "input" in features:
            return AlpacaPreprocessor(**self.kwargs)
        return ResponsePreprocessor(**self.kwargs)

    def __call__(
        self,
        dataset: DATASET_TYPE,
        *,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        strict: bool = False,
    ) -> DATASET_TYPE:
        dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)
        preprocessor = self._get_preprocessor(dataset)
        preprocessor.dataset_sample = self.dataset_sample
        return preprocessor(dataset, num_proc=num_proc, load_from_cache_file=load_from_cache_file, strict=strict)
