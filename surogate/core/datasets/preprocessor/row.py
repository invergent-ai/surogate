import os
from collections import Counter
from contextlib import contextmanager
from itertools import chain
from typing import Optional, Dict, Union, Any, List
import numpy as np
from datasets import Dataset as HfDataset
from datasets import Image
from datasets import IterableDataset as HfIterableDataset
from datasets import Sequence, Value
from surogate.core.datasets.utils import DATASET_TYPE
from surogate.utils.fs import get_cache_dir
from surogate.utils.logger import get_logger
from surogate.utils.seed import RAND_SEED

logger = get_logger()

_pair_keys = ['messages', 'images', 'videos', 'audios', 'tools', 'objects']


class MaxLengthError(ValueError):
    pass


class RowPreprocessor:
    standard_keys = _pair_keys + list(
        chain.from_iterable([f'{prefix}_{k}' for k in _pair_keys]
                            for prefix in ['rejected', 'positive', 'negative'])) + [
                        'rejected_response',
                        'label',
                        'channel',
                        'margin',
                    ]

    def __init__(
        self,
        *,
        columns: Optional[Dict[str, str]] = None,
        dataset_sample: Optional[int] = None,
        random_state: Optional[Union[np.random.RandomState, int]] = RAND_SEED,
        traceback_limit: int = 10
    ) -> None:
        self.columns = columns or {}
        self.origin_columns = self.columns.copy()
        images_keys = ['images', 'image']
        audios_keys = ['audios', 'audio']
        videos_keys = ['videos', 'video']
        for mm_type in ['images', 'audios', 'videos']:
            keys = locals()[f'{mm_type}_keys']
            for key in keys:
                self.columns[key] = mm_type
        self.traceback_limit = traceback_limit
        self._traceback_counter = 0
        self.dataset_sample = dataset_sample
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def prepare_dataset(self, dataset: DATASET_TYPE) -> DATASET_TYPE:
        return dataset

    def __call__(
            self,
            dataset: DATASET_TYPE,
            *,
            num_proc: int = 1,
            load_from_cache_file: bool = True,
            strict: bool = False,
            batch_size: Optional[int] = None,
    ) -> DATASET_TYPE:
        from ..utils import sample_dataset
        if batch_size is None:
            batch_size = 1000 if isinstance(dataset, HfDataset) else 16
        if self.dataset_sample is not None:
            dataset = sample_dataset(dataset, self.dataset_sample, True, self.random_state)

        map_kwargs = {'batched': True, 'batch_size': batch_size}
        if isinstance(dataset, HfDataset):
            map_kwargs.update({
                'num_proc': num_proc,
                'load_from_cache_file': load_from_cache_file,
            })

        # compat GRPO: The solution field will be retained.
        dataset = RowPreprocessor.get_features_dataset(dataset)
        if 'solution' in dataset.features:
            if isinstance(dataset, HfDataset) and not dataset.cache_files:
                map_kwargs['cache_file_name'] = os.path.join(get_cache_dir(), 'datasets', 'map_cache',
                                                             f'{dataset._fingerprint}.arrow')
            dataset = dataset.map(lambda x: {'__#solution': x['solution']}, **map_kwargs)
            map_kwargs.pop('cache_file_name', None)

        dataset = self.safe_rename_columns(dataset, self.origin_columns)
        dataset = self.safe_rename_columns(dataset, self.columns)
        dataset = self.prepare_dataset(dataset)
        dataset = self._cast_pil_image(dataset)
        if isinstance(dataset, HfIterableDataset):
            # fix: https://github.com/huggingface/datasets/issues/6408
            columns = {k: f'__@{k}' for k in RowPreprocessor.standard_keys if k in dataset.features}
            if columns:
                dataset = dataset.rename_columns(columns)

        ignore_max_length_error = True if isinstance(dataset, HfDataset) and num_proc > 1 else False
        with self._patch_arrow_writer():
            try:
                if isinstance(dataset, HfDataset) and not dataset.cache_files:
                    map_kwargs['cache_file_name'] = os.path.join(get_cache_dir(), 'datasets', 'map_cache',
                                                                 f'{dataset._fingerprint}.arrow')
                dataset_mapped = dataset.map(
                    self.batched_preprocess,
                    fn_kwargs={
                        'strict': strict,
                        'ignore_max_length_error': ignore_max_length_error
                    },
                    remove_columns=list(dataset.features.keys()),
                    **map_kwargs)
            except NotImplementedError:
                pass

        if isinstance(dataset_mapped, HfDataset) and len(dataset) != len(dataset_mapped):
            logger.info(
                f'Dataset filtered, origin length: {len(dataset)}, filtered dataset length: {len(dataset_mapped)}')

        return dataset_mapped

    def batched_preprocess(
            self,
            batched_row: Dict[str, Any],
            *,
            strict: bool,
            ignore_max_length_error: bool
    ) -> Dict[str, Any]:
        batched_row = dict(batched_row)
        assert len(batched_row) > 0
        self._remove_prefix_keys(batched_row, '__@')
        rows = self.batched_to_rows(batched_row)

        # Check if this preprocessor supports batched processing
        use_batched = hasattr(self, 'preprocess_batch') and callable(getattr(self, 'preprocess_batch', None))

        new_rows = []

        if use_batched:
            # Use batched preprocessing (much faster for tokenization)
            try:
                processed_rows = self.preprocess_batch(rows)

                # Post-process each result
                for row in processed_rows:
                    try:
                        # support [row1, row2, ...]
                        if row is None:
                            row = []
                        if isinstance(row, dict):
                            row = [row]
                        for r in row:
                            self._check_objects(r)
                            self._check_rejected_response(r)
                            self._check_messages(r)
                            self._cast_mm_data(r)
                        new_rows += row
                    except Exception as e:
                        if strict:
                            logger.warning('To avoid errors, you can pass `strict=False`.')
                            raise
                        if isinstance(e, MaxLengthError) and ignore_max_length_error:
                            pass
                        elif self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
                            import traceback
                            logger.info(traceback.format_exc())
                            logger.warning('👆👆👆There are errors in the dataset, the data will be deleted')
                            self._traceback_counter += 1
            except Exception as e:
                # If batch processing fails entirely, fall back to row-by-row
                logger.warning(f'Batched preprocessing failed ({e}), falling back to row-by-row processing')
                use_batched = False

        if not use_batched:
            # Fall back to row-by-row preprocessing (original behavior)
            for row in rows:
                try:
                    row = self.preprocess(row)
                    # support [row1, row2, ...]
                    if row is None:
                        row = []
                    if isinstance(row, dict):
                        row = [row]
                    for r in row:
                        self._check_objects(r)
                        self._check_rejected_response(r)
                        self._check_messages(r)
                        self._cast_mm_data(r)
                except Exception as e:
                    if strict:
                        logger.warning('To avoid errors, you can pass `strict=False`.')
                        raise
                    if isinstance(e, MaxLengthError) and ignore_max_length_error:
                        pass
                    elif self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
                        import traceback
                        logger.info(traceback.format_exc())
                        logger.warning('👆👆👆There are errors in the dataset, the data will be deleted')
                        self._traceback_counter += 1
                    row = []
                new_rows += row

        res = self.rows_to_batched(new_rows)
        self._remove_prefix_keys(res, '__#')  # compat GRPO
        if len(res) == 0:
            res['messages'] = []

        return res

    def _cast_pil_image(self, dataset):
        features = dataset.features
        for col in ['images', 'rejected_images']:
            if (col in features and isinstance(features[col], Image) and getattr(features[col], 'decode', False)):
                dataset = dataset.cast_column(col, Image(decode=False))
        return dataset

    @staticmethod
    @contextmanager
    def _patch_arrow_writer():
        # fix AI-ModelScope/ms_agent_for_agentfabric:all
        from datasets.arrow_writer import ArrowWriter

        def _new_init(self, schema=None, features=None, *args, **kwargs):

            if features is not None:
                messages_feature = [{
                    'role': Value(dtype='string'),
                    'content': Value(dtype='string'),
                }]
                messages_feature_with_loss = [{
                    'role': Value(dtype='string'),
                    'content': Value(dtype='string'),
                    'loss': Value(dtype='float64'),
                }]
                features['messages'] = messages_feature_with_loss
                features['rejected_messages'] = messages_feature_with_loss
                features['positive_messages'] = [messages_feature]
                features['negative_messages'] = [messages_feature]
                features['images'] = [{'bytes': Value(dtype='binary'), 'path': Value(dtype='string')}]
                features['objects'] = {
                    'ref': Sequence(feature=Value(dtype='string'), length=-1),
                    'bbox': Sequence(feature=Sequence(feature=Value(dtype='float64'), length=-1), length=-1),
                    'bbox_type': Value(dtype='string'),
                    'image_id': Sequence(feature=Value(dtype='int64'), length=-1),
                }
            ArrowWriter.__origin_init__(self, schema, features, *args, **kwargs)

        ArrowWriter.__origin_init__ = ArrowWriter.__init__
        ArrowWriter.__init__ = _new_init
        try:
            yield
        finally:
            ArrowWriter.__init__ = ArrowWriter.__origin_init__
            del ArrowWriter.__origin_init__

    @staticmethod
    def batched_to_rows(batched_row: Dict[str, Any]):
        keys = list(batched_row.keys())
        batch_size = len(batched_row[keys[0]])
        return [{key: batched_row[key][i] for key in keys} for i in range(batch_size)]

    @staticmethod
    def _remove_prefix_keys(row, prefix: str):
        for k in list(row.keys()):
            if k.startswith(prefix):
                new_k = k[len(prefix):]
                new_v = row.pop(k)
                if new_k not in row:
                    row[new_k] = new_v

    @staticmethod
    def _check_objects(row):
        objects = row.get('objects')
        if objects is None:
            return
        new_objects = {}
        # Ensure the order
        for k in ['ref', 'bbox', 'bbox_type', 'image_id']:
            if k in objects.keys():
                new_objects[k] = objects[k]
        row['objects'] = new_objects
        bbox = new_objects['bbox']

        # check bbox
        for box in bbox:
            assert len(box) in {2, 4}, f'len(box): {len(box)}'
            if len(box) == 2:
                continue
            if box[0] > box[2]:
                box[0], box[2] = box[2], box[0]
            if box[1] > box[3]:
                box[1], box[3] = box[3], box[1]

    @staticmethod
    def _check_rejected_response(row: Dict[str, Any]) -> None:
        if 'rejected_response' in row:
            messages = row['messages']
            rejected_response = row['rejected_response']
            if (rejected_response is None
                    or isinstance(rejected_response, str) and rejected_response == messages[-1]['content']):
                raise ValueError(f'rejected_response: {rejected_response}')

    @staticmethod
    def _check_messages(row: Dict[str, Any]) -> None:
        if 'messages' not in row:
            return
        messages = row['messages']
        assert len(messages) > 0, f'messages: {messages}'
        for message in messages:
            keys = set(message.keys()) - {'role', 'content', 'loss'}
            for key in keys:
                message.pop(key)

        for message in messages:
            role, content = message['role'], message['content']
            # The terms 'tool' and 'tool_response' have the same meaning, ensuring compatibility.
            assert role in {'system', 'user', 'tool_call', 'tool_response', 'tool', 'assistant'}, f'message: {message}'
            assert content is not None, f'message: {message}'

    @staticmethod
    def _cast_mm_data(row: Dict[str, Any]) -> None:
        for key in ['images', 'rejected_images']:
            images = row.get(key, None)
            if images is None:
                continue

            if isinstance(images, str) or (isinstance(images, list) and images and isinstance(images[0], str)):
                if isinstance(images, str):
                    images = [images]
                for i, image in enumerate(images):
                    images[i] = {'bytes': None, 'path': image}
                row[key] = images
            elif isinstance(images, dict):
                row[key] = [images]

        for key in ['videos', 'audios']:
            mm_data = row.get(key)
            if mm_data is None:
                continue
            elif isinstance(mm_data, str):
                row[key] = [mm_data]

    @staticmethod
    def rows_to_batched(rows: List[Dict[str, Any]]):
        batched = {}
        for i, row in enumerate(rows):
            for k, v in row.items():
                if k not in batched:
                    batched[k] = [None] * i
                batched[k].append(v)
            # Make all the lengths of v the same.
            for k in set(batched.keys()) - set(row.keys()):
                batched[k].append(None)
        return batched

    @staticmethod
    def get_features_dataset(dataset: DATASET_TYPE) -> DATASET_TYPE:
        if dataset.features is None:
            assert isinstance(dataset, HfIterableDataset)
            dataset = dataset._resolve_features()
        return dataset

    @staticmethod
    def safe_rename_columns(dataset, columns):
        dataset = RowPreprocessor.get_features_dataset(dataset)
        columns_keys = {k.lower(): k for k in dataset.features.keys()}  # lower -> lower/upper
        safe_columns = {columns_keys[k.lower()]: v for k, v in columns.items() if k.lower() in columns_keys}

        counter = Counter(safe_columns.values())
        for k, new_k in list(safe_columns.items()):
            if counter[new_k] > 1:
                # For example, if "response" and "answer" match, then no processing is done.
                safe_columns.pop(k)
                continue

        # e.g. Keep {'query': 'query'} to ensure that the query has the highest priority.
        safe_columns = {k: v for k, v in safe_columns.items() if k != v}
        if safe_columns:
            dataset = dataset.rename_columns(safe_columns)

        return dataset
