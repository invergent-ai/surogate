import unittest
import pytest

from transformers import AutoTokenizer
from surogate.datasets.datasets import load_datasets
from surogate.utils.dict import DictDefault
from datasets import Dataset

class TestDatasets:
    @pytest.fixture
    def conversationDatasetFixture(self):
        yield Dataset.from_list([
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there! How can I assist you today?"}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What's the weather like?"},
                    {"role": "assistant", "content": "It's sunny and warm today."}
                ]
            }
        ])

    def test_load_conversation_dataset(self):
        cfg = DictDefault({
            "sequence_len": 2048,
            "tokenizer": "Qwen/Qwen3-0.6B",
            "datasets": [
                {
                    "path": "invergent/self-cognition-qwen3",
                    "split": "train",
                    "samples": 128,
                    "type": "conversation",
                    "messages_field": "messages",
                    "message_property_mappings": {
                        "role": "role",
                        "content": "content"
                    }
                }
            ]
        })
        dataset = load_datasets(cfg)
        assert len(dataset) == 50
        assert "input_ids" in dataset.features

    def test_load_instruction_dataset(self):
        cfg = DictDefault({
            "model": "Qwen/Qwen3-0.6B",
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

    def test_extract_chat_template_variables(self):
        from surogate.datasets.prompters import get_chat_template_msg_variables

