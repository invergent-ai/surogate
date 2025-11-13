import pytest
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from surogate.datasets.wrapper import get_dataset_wrapper
from surogate.utils.dict import DictDefault
from surogate.utils.schema.datasets import ConversationDataset, InstructionDataset


@pytest.fixture(scope="class")
def tokenizer_with_chat_template():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


@pytest.fixture(scope="class")
def tokenizer_without_chat_template():
    return AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")


@pytest.fixture(scope="class")
def conversation_dataset_fixture():
    yield Dataset.from_list([
        {
            "messages": [
                {"role": "system", "content": "You are a technical support bot."},
                {"role": "user", "content": "My internet is not working."},
                {"role": "assistant",
                 "content": "I can help with that. Is the light on your router blinking or solid?"},
                {"role": "user", "content": "It is blinking red."},
                {"role": "assistant",
                 "content": "A blinking red light usually indicates a connection error. Please try restarting it."}
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


class TestDatasets:
    def test_load_conversation_dataset(self, tokenizer_with_chat_template: PreTrainedTokenizer,
                                       conversation_dataset_fixture):
        cfg = DictDefault({
            "sequence_len": 2048,
            "datasets": [
                {
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
        dataset = get_dataset_wrapper(
            cfg,
            ConversationDataset(**cfg.get('datasets')[0]),
            tokenizer_with_chat_template,
            conversation_dataset_fixture
        )
        assert len(dataset) == 2
        assert "input_ids" in dataset.features

    @pytest.fixture
    def instruction_dataset_fixture(self):
        yield Dataset.from_list([
            {
                "question": "Translate the following English text to French:",
                "input": "Hello, how are you?",
                "answer": "Bonjour, comment ça va?"
            },
            {
                "question": "Name the capital of France.",
                "answer": "Paris"
            }
        ])

    def test_load_instruction_dataset_with_chat_template(self, tokenizer_with_chat_template: PreTrainedTokenizer,
                                                         instruction_dataset_fixture):
        cfg = DictDefault({
            "sequence_len": 2048,
            "datasets": [
                {
                    "samples": 128,
                    "type": "instruction",
                    "instruction_field": "question",
                    "output_field": "answer",
                    "input_field": "input"
                }
            ]
        })

        dataset = get_dataset_wrapper(
            cfg,
            InstructionDataset(**cfg.get('datasets')[0]),
            tokenizer_with_chat_template,
            instruction_dataset_fixture
        )
        assert len(dataset) == 2
        assert "input_ids" in dataset.features

    def test_load_instruction_dataset_no_chat_template(self, tokenizer_without_chat_template: PreTrainedTokenizer,
                                                       instruction_dataset_fixture):
        cfg = DictDefault({
            "sequence_len": 2048,
            "datasets": [
                {
                    "samples": 128,
                    "type": "instruction",
                    "instruction_field": "question",
                    "output_field": "answer",
                    "input_field": "input"
                }
            ]
        })

        dataset = get_dataset_wrapper(
            cfg,
            InstructionDataset(**cfg.get('datasets')[0]),
            tokenizer_without_chat_template,
            instruction_dataset_fixture
        )
        assert len(dataset) == 2
        assert "input_ids" in dataset.features
