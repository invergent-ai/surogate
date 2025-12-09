from collections import defaultdict
from typing import Any, Dict, Literal

from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from surogate.core.model.chat_templates.processor import get_chat_template_processor
from surogate.core.model.registry import ModelTemplate
from surogate.utils.logger import get_logger

logger = get_logger()


class PromptTokenizingStrategy:
    def __init__(
            self,
            tokenizer,
            mode: Literal['pt', 'vllm', 'sglang', 'train', 'rlhf', 'kto', 'gkd'],
            task: Literal['causal_lm', 'seq_cls', 'embedding', 'prm', 'reranker', 'generative_reranker'] = "causal_lm",
    ):
        """
        Args:
            tokenizer: The tokenizer to use for tokenizing prompts.
            mode: The mode for the template encoding:
             - 'pt', 'vllm', 'sglang' are inference modes
            - ''train', 'rlhf', 'kto', 'gkd' are training modes
            task: The type of task for which the prompt is being tokenized.
        """
        model_template: ModelTemplate = getattr(tokenizer, "model_template")
        self.template = get_chat_template_processor(model_template.chat_template, tokenizer)
        self.template.set_mode(mode)
        self.template.task_type = task
        if self.template is None:
            raise ValueError(f"Chat Template not found for the architecture {model_template.model_arch}.")

        self.tokenizer: PreTrainedTokenizerBase = tokenizer

    @property
    def supports_batched(self):
        return True

    def is_prompt_batched(self, prompt: dict[str, Any]) -> bool:
        try:
            return all(isinstance(v, list) for v in prompt.values()) and all(
                isinstance(v, list) for v in prompt['messages']
            )
        except KeyError:
            return False

    def tokenize_prompt(self, prompt):
        if not self.is_prompt_batched(prompt) or not self.supports_batched:
            return self._tokenize_single_prompt(prompt)

        res = defaultdict(lambda: [])
        feature_names = list(prompt.keys())
        # Process each prompt individually
        for row in zip(*prompt.values(), strict=False):
            tokenized_prompt = self._tokenize_single_prompt(
                dict(zip(feature_names, row, strict=False))
            )
            for key, val in tokenized_prompt.items():
                res[key].append(val)

        if not res:
            return {}

        return dict(res)

    def _tokenize_single_prompt(self, prompt: dict) -> Dict[str, Any]:
        return self.template.encode(prompt)


class TokenizedPromptDataset(Dataset):
    def __init__(
            self,
            prompt_tokenizer: PromptTokenizingStrategy,
            dataset: Dataset,
            process_count: int | None = None,
            keep_in_memory: bool | None = False,
            **kwargs,
    ):
        self.prompt_tokenizer = prompt_tokenizer
        self.process_count = process_count
        self.keep_in_memory = keep_in_memory
        super().__init__(
            self.process(dataset).data,
            **kwargs,
        )

    def process(self, dataset):
        features = dataset.features.keys()
        map_kwargs = {}
        if self.prompt_tokenizer.supports_batched:
            map_kwargs["batched"] = True
            map_kwargs["batch_size"] = 1_000

        return dataset.filter(
            drop_empty_rows,
            num_proc=self.process_count,
            keep_in_memory=self.keep_in_memory,
        ).map(
            self.prompt_tokenizer.tokenize_prompt,
            num_proc=self.process_count,
            remove_columns=features,
            keep_in_memory=self.keep_in_memory,
            desc="Tokenizing Prompts",
            **map_kwargs,
        )


def tokenize_dataset(
        prompt_tokenizer: PromptTokenizingStrategy,
        dataset: Dataset | IterableDataset,
        **kwargs,
) -> Dataset | IterableDataset:
    if isinstance(dataset, IterableDataset):
        map_kwargs = {}
        if prompt_tokenizer.supports_batched:
            map_kwargs["batched"] = True
        features = list(dataset.features.keys())
        return dataset.filter(
            drop_empty_rows,
            **map_kwargs
        ).map(
            prompt_tokenizer.tokenize_prompt,
            remove_columns=features,
            **map_kwargs,
        )

    return TokenizedPromptDataset(prompt_tokenizer, dataset, **kwargs)

def drop_empty_rows(row: dict[str, Any]) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        # invalid or empty messages
        return False

    seen_user = False
    for turn in messages:
        if turn.get("role") == "user":
            seen_user = True
            content = turn.get("content")

            # Handle only simple string content here; extend if needed for multimodal
            if content is None:
                return False

            if not isinstance(content, str):
                # if the row can have non-string content, decide what to do here
                return False

            if not content.strip():
                # empty user turn → drop conversation
                return False

    # Require at least one user turn
    return seen_user