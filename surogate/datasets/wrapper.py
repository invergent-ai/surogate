import os
from typing import Dict, Any, Optional

from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase

from surogate.datasets.prompters import Prompter, ChatTemplatePrompter
from surogate.datasets.strategies import ChatTemplateStrategy, PromptTokenizingStrategy
from surogate.utils.dict import DictDefault
from surogate.utils.schema.datasets import ConversationDataset, InstructionDataset, TextDataset, BaseDataset
from surogate.utils.schema.enums import SurogateDatasetType, ChatTemplateType

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

        if (
            hasattr(self.prompt_tokenizer, "filter_rows")
            and self.prompt_tokenizer.filter_rows
        ):
            dataset = dataset.filter(
                self.prompt_tokenizer.filter_rows,
                num_proc=self.process_count,
                desc="Strategy Filtering Rows",
            )

        return dataset.map(
            self.prompt_tokenizer.tokenize_prompt,
            num_proc=self.process_count,
            remove_columns=features,
            keep_in_memory=self.keep_in_memory,
            desc="Tokenizing Prompts",
            **map_kwargs,
        )


def get_dataset_wrapper(
        cfg: DictDefault,
        dataset_config: ConversationDataset | InstructionDataset | TextDataset,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset | IterableDataset,
) -> Dataset | IterableDataset:
    dataset_kwargs: dict[str, Any] = {
        "process_count": get_default_process_count(),
        "keep_in_memory": False
    }

    if dataset_config.type == SurogateDatasetType.conversation:
        chat_template_string = get_chat_template_from_config(
            ds_cfg=dataset_config, tokenizer=tokenizer
        )
        prompter = ChatTemplatePrompter(
            tokenizer=tokenizer,
            chat_template=chat_template_string,
            message_property_mappings=dataset_config.message_property_mappings or {},
            messages_field=dataset_config.messages_field,
            system_field=dataset_config.system_field,
            tools_field=dataset_config.tools_field,
            sequence_len=cfg.get('sequence_len'),
        )
        dataset_strategy = ChatTemplateStrategy(prompter, tokenizer, sequence_len=cfg.get('sequence_len'))
        return wrap_dataset_for_tokenized_prompt(dataset_strategy, dataset, **dataset_kwargs)
    elif dataset_config.type == SurogateDatasetType.instruction:
        chat_template_string = get_chat_template_from_config(
            ds_cfg=dataset_config, tokenizer=tokenizer
        )

    elif dataset_config.type == SurogateDatasetType.text:
        pass
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_config.type}")


def get_chat_template_from_config(
        ds_cfg: ConversationDataset | None = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
) -> str:
    if ds_cfg.chat_template == ChatTemplateType.tokenizer_default:
        if not tokenizer:
            raise ValueError(
                f"`tokenizer` cannot be None when chat_template choice is {ChatTemplateType.tokenizer_default}"
            )
        if not tokenizer.chat_template:
            raise ValueError(
                f"`chat_template choice is {ChatTemplateType.tokenizer_default} but tokenizer's chat_template is null. "
                f"Please add a chat_template in tokenizer config"
            )
        return tokenizer.chat_template
    elif ds_cfg.chat_template == ChatTemplateType.jinja:
        jinja_template = ds_cfg.chat_template_jinja
        if not jinja_template:
            raise ValueError("Jinja template path must be provided for 'jinja' chat template type.")

        if os.path.exists(jinja_template) and os.path.isfile(jinja_template):
            with open(jinja_template, "r", encoding="utf-8") as file:
                chat_template = file.read()
            return chat_template
        else:
            raise FileNotFoundError(f"Jinja template file not found: {jinja_template}")
    else:
        raise ValueError(f"Unsupported chat template type: {ds_cfg.chat_template}")


def wrap_dataset_for_tokenized_prompt(
        prompt_tokenizer: PromptTokenizingStrategy,
        dataset: Dataset | IterableDataset,
        **kwargs,
) -> Dataset | IterableDataset:
    if isinstance(dataset, IterableDataset):
        map_kwargs = {}
        if prompt_tokenizer.supports_batched:
            map_kwargs["batched"] = True
        features = list(dataset.features.keys())
        return dataset.map(
            prompt_tokenizer.tokenize_prompt,
            remove_columns=features,
            **map_kwargs,
        )
    return TokenizedPromptDataset(prompt_tokenizer, dataset, **kwargs)



def get_default_process_count():
    if axolotl_dataset_processes := os.environ.get("SUROGATE_DATASET_PROCESSES"):
        return int(axolotl_dataset_processes)
    if runpod_cpu_count := os.environ.get("RUNPOD_CPU_COUNT"):
        return int(runpod_cpu_count)
    return os.cpu_count()
