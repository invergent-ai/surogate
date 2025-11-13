import unittest

from transformers import AutoTokenizer

from surogate.datasets.datasets import load_datasets
from surogate.utils.dict import DictDefault


class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def test_load_dataset(self):
        cfg = DictDefault({
            "sequence_len": 2048,
            "datasets": [
                {
                    "path": "invergent/self-cognition",
                    "split": "train",
                    "samples": 128,
                    "type": "instruction",
                    "instruction_field": "question",
                    "output_field": "answer"
                }
            ]
        })

        dataset = load_datasets(cfg)
        assert len(dataset) == 1
        assert "input_ids" in dataset.features