from pydantic import BaseModel, Field

from surogate.utils.schema.datasets import BaseDataset


class SpecialTokensConfig(BaseModel):
    bos_token: str | None = None
    eos_token: str | None = None
    pad_token: str | None = None
    unk_token: str | None = None
    additional_special_tokens: list[str] | None = None

class SurogateBaseConfig(BaseModel):
    model_config = {"populate_by_name": True}

    seed: int | None = Field(
        default=None, json_schema_extra={"description": "Random seed for reproducibility"}
    )

    model: str = Field(
        json_schema_extra={
            "description": "HuggingFace model name or local path"
        }
    )
    tokenizer: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Path to a tokenizer config file if you want to use a custom tokenizer"
        },
    )

    special_tokens: SpecialTokensConfig | None = Field(
        default=None,
        json_schema_extra={
            "description": "Add or change special tokens. If you add tokens here, you don't need to add them to the `tokens` list."
        },
    )

    tokens: list[str] | None = Field(
        default=None,
        json_schema_extra={"description": "Add extra tokens to the tokenizer"},
    )

    datasets: list[BaseDataset] = Field(
        default=None,
        json_schema_extra={
            "description": "List of datasets to use for the task"
        },
    )



