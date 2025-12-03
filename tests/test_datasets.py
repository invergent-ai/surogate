import pytest
from datasets import Dataset
from transformers import PreTrainedTokenizer

from surogate.datasets.datasets import wrap_dataset
from surogate.datasets.tokenization import tokenize_dataset, PromptTokenizingStrategy
from surogate.utils.dict import DictDefault
from surogate.loaders.loader import get_model_and_tokenizer


@pytest.fixture(scope="class")
def tokenizer_with_chat_template():
    _, tokenizer = get_model_and_tokenizer("Qwen/Qwen3-0.6B", load_model=False)
    return tokenizer


@pytest.fixture(scope="class")
def tokenizer_without_chat_template():
    _, tokenizer = get_model_and_tokenizer("unsloth/Llama-3.2-1B", load_model=False)
    return tokenizer


class TestDatasets:
    @pytest.fixture
    def conversation_dataset_fixture(self):
        yield Dataset.from_list([
            {
                "messages": [
                    {"role": "system", "content": "You are a technical support bot."},
                    {"role": "user", "content": "My internet is not working."},
                    {"role": "assistant",
                     "content": "I can help with that. Is the light on your router blinking or solid?"},
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What's the weather like?"},
                    {"role": "assistant", "content": "It's sunny and warm today."},
                    {"role": "user", "content": "What about tomorrow?"},
                    {"role": "assistant", "content": "Tomorrow will be also sunny."}
                ]
            }
        ])

    def test_load_conversation_dataset(
            self,
            conversation_dataset_fixture,
            tokenizer_with_chat_template,
            tokenizer_without_chat_template
    ):
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
        wrapped_dataset = wrap_dataset(
            cfg,
            ConversationDataset(**cfg.get('datasets')[0]),
            conversation_dataset_fixture
        )
        assert len(wrapped_dataset) == 2

        dataset = tokenize_dataset(
            PromptTokenizingStrategy(
                tokenizer=tokenizer_with_chat_template,
                mode='pt',
            ),
            wrapped_dataset,
        )

        assert len(dataset) == 2
        assert 'input_ids' in dataset.column_names

        dataset = tokenize_dataset(
            PromptTokenizingStrategy(
                tokenizer=tokenizer_without_chat_template,
                mode='pt',
            ),
            wrapped_dataset,
        )

        assert len(dataset) == 2
        assert 'input_ids' in dataset.column_names

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

    def test_load_instruction_dataset(
            self,
            tokenizer_with_chat_template: PreTrainedTokenizer,
            tokenizer_without_chat_template: PreTrainedTokenizer,
            instruction_dataset_fixture
    ):
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

        wrapped_dataset = wrap_dataset(
            cfg,
            InstructionDataset(**cfg.get('datasets')[0]),
            instruction_dataset_fixture
        )

        assert len(wrapped_dataset) == 2

        dataset = tokenize_dataset(
            PromptTokenizingStrategy(
                tokenizer=tokenizer_with_chat_template,
                mode='pt',
            ),
            wrapped_dataset,
        )

        assert len(dataset) == 2
        assert 'input_ids' in dataset.column_names

        dataset = tokenize_dataset(
            PromptTokenizingStrategy(
                tokenizer=tokenizer_without_chat_template,
                mode='pt',
            ),
            wrapped_dataset,
        )

        assert len(dataset) == 2
        assert 'input_ids' in dataset.column_names
