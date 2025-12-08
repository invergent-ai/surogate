from typing import Optional
import multiprocessing as mp
import torch
from torch.utils.data import Dataset, IterableDataset

from surogate.core.datasets.preprocessor.row import MaxLengthError
from surogate.core.datasets.progress import create_hfhub_tqdm
from surogate.utils.dist import is_master, is_dist


class PackingDataset(Dataset):

    def __init__(
            self,
            template,
            dataset,
            num_proc: int = 1,
            *,
            strict: bool = False,
            load_from_cache_file: bool = True,
            packing_length: Optional[int] = None,
            **kwargs,
    ):
        template.packing = True
        template.padding_free = True  # TODO: remove
        self.template = template
        self.dataset = dataset
        self.num_proc = num_proc
        self.strict = strict
        self.load_from_cache_file = load_from_cache_file
        self.packing_length = packing_length or self.template.max_length
        self.workers = []
        self.packed_idx, self.packed_length = self.create_packed_idx() if is_master() else (None, None)
        if torch.distributed.is_initialized() and is_dist():
            obj_list = [(self.packed_idx, self.packed_length)]
            torch.distributed.broadcast_object_list(obj_list)
            self.packed_idx, self.packed_length = obj_list[0]

    def create_packed_idx(self):
        lengths = self.dataset['length']
        data = [(i, length) for i, length in enumerate(lengths)]
        i = 0
        PACKING_BATCH_SIZE = 1000
        input_data, packed_idx, packed_length = [], [], []
        hf_tqdm = create_hfhub_tqdm("Packing dataset:")
        with hf_tqdm(total=len(data), dynamic_ncols=True, desc='Packing: ') as prog_bar:
            while True:
                new_data = data[i:i + PACKING_BATCH_SIZE]
                input_data += new_data
                prog_bar.update(len(new_data))
                if not input_data:
                    break
                i += PACKING_BATCH_SIZE
                is_finished = i >= len(data)
                sequences, input_data = calculate_matched_group(
                    self.template, input_data, self.packing_length, is_finished=is_finished)
                packed_idx += [[x[0] for x in seq] for seq in sequences]
                packed_length += [sum(x[1] for x in seq) for seq in sequences]
        return packed_idx, packed_length

    def __getitem__(self, index):
        sequence = self.packed_idx[index]
        row = [self.dataset[i] for i in sequence]
        return row

    def __len__(self):
        return len(self.packed_idx)


class IterablePackingDataset(IterableDataset):

    def __init__(
            self,
            template,
            dataset,
            num_proc: int = 1,
            *,
            packing_interval: int = 128,
            packing_length: Optional[int] = None,
            strict: bool = False,
            cyclic: bool = False,
            **kwargs,
    ):
        template.packing = True
        template.padding_free = True  # TODO: remove
        self.template = template
        self.dataset = dataset
        self.num_proc = num_proc
        self.strict = strict
        self.packing_length = packing_length or self.template.max_length

        self.packing_interval = packing_interval
        self._in_queue = mp.Queue()
        self._out_queue = mp.Queue()
        self.workers = []
        self.cyclic = cyclic
        for _ in range(self.num_proc):
            worker = mp.Process(target=self._processor, daemon=True)
            worker.start()
            self.workers.append(worker)

    def _processor(self):
        while True:
            i, data = self._in_queue.get()
            encoded_data = {}
            try:
                encoded_data = self.template.encode(data, return_length=True)
            except Exception as e:
                if self.strict and not isinstance(e, MaxLengthError):
                    raise
            self._out_queue.put((i, encoded_data))

    def _put_data_in_queue(self, iterator) -> int:
        for i in range(self.packing_interval):
            try:
                data = next(iterator)
            except StopIteration:
                return i
            self._in_queue.put((i, data))
        return i + 1

    def _fetch_data_out_queue(self, last_res, num_samples):
        res = [None] * num_samples
        for _ in range(num_samples):
            i, data = self._out_queue.get()
            if not data:
                continue
            res[i] = (data, len(data['input_ids']))
        res = [data for data in res if data]
        last_res += res
        return last_res

    @staticmethod
    def cyclic_iter(iterable):
        while True:
            for x in iterable:
                yield x

    def __iter__(self):
        try:
            next(iter(self.dataset))
        except StopIteration:
            return

        if self.cyclic:
            iterator = self.cyclic_iter(self.dataset)
        else:
            iterator = iter(self.dataset)
        data = []
        while True:
            num_samples = self._put_data_in_queue(iterator)
            finished = num_samples != self.packing_interval
            data = self._fetch_data_out_queue(data, num_samples)
            sequences, data = calculate_matched_group(self.template, data, self.packing_length, is_finished=finished)
            res = []
            for row in sequences:
                res.append([r[0] for r in row])
            yield from res
            if finished:
                break


def calculate_matched_group(template, sequences, packing_length: int, is_finished: bool = True):
    if len(sequences) == 0:
        return [], []
    # https://arxiv.org/pdf/2404.10830
    import binpacking
    sequences = binpacking.to_constant_volume(sequences, packing_length, weight_pos=1)
    if sequences and not is_finished:
        sequences, ret_sequences = sequences[:-1], sequences[-1]
    else:
        ret_sequences = []
    return sequences, ret_sequences